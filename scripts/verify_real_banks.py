#!/usr/bin/env python
"""TDD acceptance-test runner for real (non-synthetic) KV banks.

Runs three checks per page_key:

  Check 1 — Injection changes logits
      Load the Llama model, load a bank from the manifest, inject it via MI
      attention, and verify that bank-injected logits diverge from baseline
      (KL > 1e-3).  Also verifies that clear_bank() restores the bit-exact
      baseline logits.

  Check 2 — Bank byte contract
      Inspect every .bin file referenced in the manifest and confirm:
        * shape is [8, num_slots, 128] float32 (C-order)
        * num_slots is identical across all 4 selected layers
        * the manifest entry is NOT tagged ``synthetic: true``

  Check 3 — Efficacy threshold
      Run the efficacy harness (KL-to-DOM / KL-to-empty) and assert that the
      roll-up score meets the minimum threshold (default 0.1).

Usage:
    python scripts/verify_real_banks.py
    python scripts/verify_real_banks.py --page-key hn:front --page-key hn:item
    python scripts/verify_real_banks.py --threshold 0.15 --skip-contract
    python scripts/verify_real_banks.py --banks-dir path/to/banks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — mirrors the project convention so that shared packages and
# app-internal modules are importable without editable installs.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "shared-py"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "inference-engine" / "src"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "bank-compiler" / "src"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "bank-compiler"))

# ---------------------------------------------------------------------------
# ANSI helpers for color-coded output
# ---------------------------------------------------------------------------
_GREEN = "\x1b[92m"
_RED = "\x1b[91m"
_YELLOW = "\x1b[93m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"


def _pass(msg: str) -> str:
    return f"{_GREEN}PASS{_RESET} {msg}"


def _fail(msg: str) -> str:
    return f"{_RED}FAIL{_RESET} {msg}"


def _skip(msg: str) -> str:
    return f"{_YELLOW}SKIP{_RESET} {msg}"


def _header(text: str) -> str:
    return f"\n{_BOLD}{'=' * 70}{_RESET}\n{_BOLD}  {text}{_RESET}\n{_BOLD}{'=' * 70}{_RESET}"


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
NUM_KV_HEADS = 8
HEAD_DIM = 128
FLOAT32_BYTES = 4
SELECTED_LAYERS = [8, 12, 16, 20]
DEFAULT_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Model bootstrap
# ---------------------------------------------------------------------------
def _load_backend(banks_dir: str):
    """Load the Llama-3.1-8B model with MI attention installed.

    Returns a ``LlamaBackend`` whose ``.model`` and ``.tokenizer`` can be
    passed directly to the injection checks and the efficacy harness.
    """
    import torch
    from inference_engine.config import Settings
    from inference_engine.engine import LlamaBackend

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the model checks.")

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Llama-3.1-8B-Instruct is a gated model."
        )

    settings = Settings(
        model_id=os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"),
        clickhouse_url="http://localhost:8123",
        banks_dir=banks_dir,
        port=8000,
        hf_token=os.environ["HF_TOKEN"],
    )
    print("[setup] Loading Llama-3.1-8B-Instruct (this may take a minute) ...")
    backend = LlamaBackend.load(settings)
    print("[setup] Model ready.")
    return backend


# ======================================================================
# CHECK 1 — Injection changes logits
# ======================================================================
def check_injection_changes_logits(
    page_key: str,
    banks_dir: str,
    backend,
) -> bool:
    """Verify that bank injection measurably changes logits and clears cleanly.

    Logic mirrors ``tests/integration/test_04_injection_changes_logits.py``
    exactly.
    """
    import torch
    import torch.nn.functional as F
    from inference_engine.bank_registry import BankRegistry
    from inference_engine.engine import build_messages
    from inference_engine.mi_attention import clear_banks, set_banks

    # --- Load bank ----------------------------------------------------------
    reg = BankRegistry.from_manifest_dir(banks_dir)
    layer_banks = reg.get(page_key)
    if layer_banks is None:
        print(f"  Bank {page_key!r} not found in manifest — cannot test injection.")
        return False

    # --- Build prompt -------------------------------------------------------
    messages = build_messages(
        "Find the top AI story",
        "https://news.ycombinator.com/",
        history=[],
        dom_text=None,
        latent_context=False,
    )
    input_ids = backend.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(backend.model.device)

    # --- Baseline logits (no bank) ------------------------------------------
    clear_banks(backend.model)
    with torch.no_grad():
        out_baseline = backend.model(input_ids)
    logits_baseline = out_baseline.logits[0, -1, :].float()

    # --- MI logits (bank injected) ------------------------------------------
    set_banks(backend.model, layer_banks)
    with torch.no_grad():
        out_mi = backend.model(input_ids)
    logits_mi = out_mi.logits[0, -1, :].float()

    # --- Restored logits (bank cleared) -------------------------------------
    clear_banks(backend.model)
    with torch.no_grad():
        out_restored = backend.model(input_ids)
    logits_restored = out_restored.logits[0, -1, :].float()

    # --- Assertions ---------------------------------------------------------
    kl = F.kl_div(
        F.log_softmax(logits_mi, dim=-1),
        F.softmax(logits_baseline, dim=-1),
        reduction="sum",
    ).item()

    if kl <= 1e-3:
        print(f"  KL divergence {kl:.6f} <= 1e-3 — bank has no measurable effect.")
        return False

    if not torch.allclose(logits_restored, logits_baseline, atol=1e-5):
        print("  clear_bank() did NOT restore bit-exact baseline logits.")
        return False

    print(f"  KL(mi || baseline) = {kl:.4f}  OK clear_bank restores bit-exact baseline")
    return True


# ======================================================================
# CHECK 2 — Bank byte contract
# ======================================================================
def check_bank_contract(
    page_key: str,
    banks_dir: str,
) -> bool:
    """Verify every .bin file respects the binary contract.

    Checks:
      - Each .bin file byte-size is exactly 8 * num_slots * 128 * 4.
      - num_slots is identical across all 4 selected layers.
      - The manifest entry is NOT tagged ``"synthetic": true``.
    """
    from agentinception_shared import bank_io

    # --- Read manifest entry ------------------------------------------------
    manifest = bank_io.read_manifest(banks_dir)
    entry = None
    for e in manifest.get("banks", []):
        if e.get("page_key") == page_key:
            entry = e
            break

    if entry is None:
        print(f"  No manifest entry for {page_key!r}.")
        return False

    # --- Synthetic tag ------------------------------------------------------
    if entry.get("synthetic") is True:
        print(f"  Bank {page_key!r} is tagged synthetic — not a real bank.")
        return False

    # --- Validate .bin files ------------------------------------------------
    declared_slots = int(entry.get("num_slots", 0))
    if declared_slots <= 0:
        print(f"  num_slots={declared_slots} is invalid.")
        return False

    expected_size = NUM_KV_HEADS * declared_slots * HEAD_DIM * FLOAT32_BYTES
    layers_seen: dict[int, int] = {}  # layer -> num_slots deduced from file size

    for layer_str, names in entry.get("files", {}).items():
        layer = int(layer_str)
        for kind in ("k", "v"):
            fname = names.get(kind)
            if fname is None:
                print(f"  Missing {kind!r} file for layer {layer} in manifest entry.")
                return False
            fpath = os.path.join(banks_dir, fname)
            if not os.path.isfile(fpath):
                print(f"  File not found: {fpath}")
                return False
            actual_size = os.path.getsize(fpath)
            if actual_size != expected_size:
                print(
                    f"  {fname}: expected {expected_size} bytes, got {actual_size} "
                    f"(for num_slots={declared_slots})."
                )
                return False
        layers_seen[layer] = declared_slots

    # --- num_slots consistency across layers --------------------------------
    if len(set(layers_seen.values())) != 1:
        print(f"  num_slots varies across layers: {layers_seen}")
        return False

    expected_layers = set(SELECTED_LAYERS)
    if set(layers_seen.keys()) != expected_layers:
        print(
            f"  Layer set mismatch: expected {sorted(expected_layers)}, "
            f"got {sorted(layers_seen.keys())}."
        )
        return False

    print(
        f"  All {len(layers_seen) * 2} .bin files match "
        f"[8, {declared_slots}, 128] float32  OK not synthetic"
    )
    return True


# ======================================================================
# CHECK 3 — Efficacy threshold
# ======================================================================
def _load_dom_text(page_key: str, dom_file: str | None) -> str:
    """Resolve DOM text for the ground-truth condition.

    Mirrors the logic in ``scripts/measure_efficacy.py``.
    """
    if dom_file:
        return Path(dom_file).read_text(encoding="utf-8")

    candidates = [
        REPO_ROOT / "demo-assets" / "snapshots" / f"{page_key.replace(':', '_')}.html",
    ]
    if page_key == "popup:demo":
        candidates.append(REPO_ROOT / "demo-assets" / "popup-page" / "index.html")
    for c in candidates:
        if c.exists():
            return c.read_text(encoding="utf-8")

    print(
        f"  WARNING: no DOM file found for {page_key}; KL-to-DOM will be NaN.",
        file=sys.stderr,
    )
    return ""


def check_efficacy(
    page_key: str,
    banks_dir: str,
    threshold: float,
    backend,
    dom_file: str | None = None,
) -> bool:
    """Run the efficacy harness and assert roll-up score >= threshold.

    Uses the same scoring path as ``scripts/measure_efficacy.py``, but reuses
    the already-loaded ``backend`` (which has MI attention installed) instead
    of loading a fresh model.
    """
    import torch
    from efficacy.harness import load_probes, score_bank

    from agentinception_shared import bank_io

    # --- Load bank data -----------------------------------------------------
    manifest = bank_io.read_manifest(banks_dir)
    entry = None
    for e in manifest.get("banks", []):
        if e.get("page_key") == page_key:
            entry = e
            break
    if entry is None:
        print(f"  No manifest entry for {page_key!r}.")
        return False

    bank_data = bank_io.load_bank(entry, banks_dir)
    bank_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_id, (k_np, v_np) in bank_data.items():
        bank_layers[layer_id] = (
            torch.from_numpy(k_np.copy()),
            torch.from_numpy(v_np.copy()),
        )

    # --- Load probes --------------------------------------------------------
    try:
        probes = load_probes(page_key)
    except FileNotFoundError:
        print(f"  No probe set for {page_key!r} — cannot measure efficacy.")
        return False
    print(f"  Loaded {len(probes)} probes for {page_key}")

    # --- DOM text -----------------------------------------------------------
    dom_text = _load_dom_text(page_key, dom_file)

    # --- Score --------------------------------------------------------------
    result = score_bank(
        model=backend.model,
        tokenizer=backend.tokenizer,
        bank_layers=bank_layers,
        probes=probes,
        dom_text=dom_text,
    )

    # --- Report -------------------------------------------------------------
    print(f"  avg KL-to-DOM:    {result['avg_kl_to_dom']:.4f}")
    print(f"  avg KL-to-empty:  {result['avg_kl_to_empty']:.4f}")
    print(f"  roll-up score:    {result['rollup_score']:.4f}  (threshold: {threshold})")

    if result["rollup_score"] != result["rollup_score"]:  # NaN check
        print("  Roll-up is NaN — missing DOM or bank data.")
        return False

    if result["rollup_score"] >= threshold:
        return True
    else:
        print(f"  Roll-up {result['rollup_score']:.4f} is below threshold {threshold}.")
        return False


# ======================================================================
# Main
# ======================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--page-key",
        action="append",
        default=None,
        help="Page key to verify (repeatable). If omitted, all banks in the "
        "manifest are tested.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum roll-up score for Check 3 (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--skip-contract",
        action="store_true",
        help="Skip Check 2 (byte contract).",
    )
    parser.add_argument(
        "--skip-logits",
        action="store_true",
        help="Skip Check 1 (injection changes logits).",
    )
    parser.add_argument(
        "--skip-efficacy",
        action="store_true",
        help="Skip Check 3 (efficacy threshold).",
    )
    parser.add_argument(
        "--banks-dir",
        default=str(REPO_ROOT / "banks"),
        help="Directory containing manifest.json + .bin files.",
    )
    parser.add_argument(
        "--dom-file",
        help="Path to DOM HTML file for Check 3 ground-truth comparison.",
    )
    args = parser.parse_args()

    banks_dir = os.path.abspath(args.banks_dir)

    # --- Resolve page keys --------------------------------------------------
    from agentinception_shared import bank_io

    manifest = bank_io.read_manifest(banks_dir)
    all_page_keys = sorted({e["page_key"] for e in manifest.get("banks", [])})

    page_keys = args.page_key if args.page_key else all_page_keys
    if not page_keys:
        print(
            "ERROR: no page keys specified and manifest contains no banks.",
            file=sys.stderr,
        )
        return 1

    # Validate page keys exist in the manifest.
    for pk in page_keys:
        if pk not in all_page_keys:
            print(
                f"ERROR: page_key {pk!r} not found in {banks_dir}/manifest.json",
                file=sys.stderr,
            )
            return 1

    # --- Determine whether model is needed ----------------------------------
    need_model = not args.skip_logits or not args.skip_efficacy
    backend = None
    if need_model:
        try:
            backend = _load_backend(banks_dir)
        except Exception as exc:
            print(f"ERROR: cannot load model: {exc}", file=sys.stderr)
            if not args.skip_logits or not args.skip_efficacy:
                return 1
            print("  Proceeding with checks that do not require the model.")

    overall_pass = True

    # --- Run checks per page_key --------------------------------------------
    for pk in page_keys:
        print(_header(f"Bank: {pk}"))

        # Check 1 — Injection changes logits
        if args.skip_logits:
            print(_skip("Check 1 — injection changes logits"))
        else:
            print("[Check 1] Injection changes logits ...")
            try:
                ok = check_injection_changes_logits(pk, banks_dir, backend)
            except Exception as exc:
                print(_fail(f"Check 1 raised: {exc}"))
                ok = False
            if ok:
                print(_pass("Check 1"))
            else:
                print(_fail("Check 1"))
            overall_pass = overall_pass and ok

        # Check 2 — Bank byte contract
        if args.skip_contract:
            print(_skip("Check 2 — bank byte contract"))
        else:
            print("[Check 2] Bank byte contract ...")
            try:
                ok = check_bank_contract(pk, banks_dir)
            except Exception as exc:
                print(_fail(f"Check 2 raised: {exc}"))
                ok = False
            if ok:
                print(_pass("Check 2"))
            else:
                print(_fail("Check 2"))
            overall_pass = overall_pass and ok

        # Check 3 — Efficacy threshold
        if args.skip_efficacy:
            print(_skip("Check 3 — efficacy threshold"))
        else:
            print("[Check 3] Efficacy threshold ...")
            try:
                ok = check_efficacy(
                    pk,
                    banks_dir,
                    args.threshold,
                    backend,
                    dom_file=args.dom_file,
                )
            except Exception as exc:
                print(_fail(f"Check 3 raised: {exc}"))
                ok = False
            if ok:
                print(_pass("Check 3"))
            else:
                print(_fail("Check 3"))
            overall_pass = overall_pass and ok

    # --- Final summary ------------------------------------------------------
    print(_header("Summary"))
    if overall_pass:
        print(_pass("All checks passed."))
    else:
        print(_fail("One or more checks failed."))

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
