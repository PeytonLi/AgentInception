"""Cumulative token metrics for the comparison chart (A3 brief task 5).

Two numbers matter for the demo, and they are *different*:

* ``kv_savings_ratio`` - the **observed** ratio ``cum_baseline / cum_visible``.
  This is what CONTRACTS s7 ``token_metrics`` carries and what the console plots;
  it grows as a run accumulates steps.
* ``structural_kv_ratio`` - the **theoretical** KV-cache ratio from the README
  headline formula ``(NUM_LAYERS * T_guidance) / (L_injected * S_bank)``. This is
  a per-page property of the bank, not of the run.

P3 brief task 3 requires the printed structural ratio to match the formula
exactly - never a fudged display value. We compute it here from real, tracked
inputs (the DOM token count baseline would have sent, the injected layer count,
and the bank's slot count) so the number is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from agentinception_shared.constants import NUM_LAYERS
except Exception:  # pragma: no cover - shared-py always present in the monorepo
    NUM_LAYERS = 32


def kv_cache_ratio(
    t_guidance: int,
    num_slots: int,
    num_injected_layers: int,
    num_layers: int = NUM_LAYERS,
) -> float:
    """README KV-cache savings ratio: (NUM_LAYERS * T_guidance) / (L * S_bank).

    ``t_guidance`` is the full DOM token count baseline would have sent this
    page, ``num_slots`` is the bank's slot count, ``num_injected_layers`` is the
    count of layers carrying the bank (4 for [8,12,16,20]). Returns ``0.0`` when
    no bank was injected (the denominator would be zero).
    """
    denom = num_injected_layers * num_slots
    if denom <= 0 or t_guidance <= 0:
        return 0.0
    return round((num_layers * t_guidance) / denom, 1)


@dataclass
class Metrics:
    cum_visible: int = 0
    cum_baseline: int = 0
    steps: int = 0
    # Structural inputs, captured from the largest bank-backed step so the
    # headline ratio reflects the worst-case (densest) page the run hit.
    peak_dom_tokens: int = 0
    bank_slots: int = 0
    injected_layers: int = 0

    def record(
        self,
        visible_tokens: int,
        baseline_tokens: int,
        *,
        dom_token_count: int | None = None,
        num_slots: int | None = None,
        num_injected_layers: int | None = None,
    ) -> None:
        self.cum_visible += int(visible_tokens)
        self.cum_baseline += int(baseline_tokens)
        self.steps += 1
        # Capture the structural inputs from the densest bank-backed step, kept
        # paired (T_guidance and S_bank from the *same* page) so the headline
        # ratio is the worst-case page's real (NUM_LAYERS*T)/(L*S).
        if num_slots and num_injected_layers and dom_token_count:
            if int(dom_token_count) > self.peak_dom_tokens:
                self.peak_dom_tokens = int(dom_token_count)
                self.bank_slots = int(num_slots)
                self.injected_layers = int(num_injected_layers)

    @property
    def kv_savings_ratio(self) -> float:
        if self.cum_visible <= 0:
            return 0.0
        return round(self.cum_baseline / self.cum_visible, 1)

    @property
    def structural_kv_ratio(self) -> float:
        """The README headline formula, computed from tracked real inputs."""
        return kv_cache_ratio(
            self.peak_dom_tokens, self.bank_slots, self.injected_layers
        )
