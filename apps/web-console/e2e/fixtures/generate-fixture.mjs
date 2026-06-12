// Regenerates `real-run.json`: a recorded sequence of CONTRACTS.md §7 events that
// mirrors a real live Hacker News run in `--mode=mi`. Viewport frames are rendered
// with Chromium (JPEG q50, 1280×720, matching the engine's frame contract) so the
// committed fixture needs no GPU and replays deterministically in CI.
//
//   node e2e/fixtures/generate-fixture.mjs
//
// The e2e test (`dashboard.spec.ts`) replays this fixture over a fake WebSocket.

import { chromium } from "@playwright/test";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));

const SHELL = (
  body,
  bg = "#f6f6ef",
) => `<!doctype html><html><head><meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: Verdana, Geneva, sans-serif; background: ${bg}; color: #000; }
  .topbar { background: #ff6600; padding: 4px 8px; display: flex; align-items: center; gap: 10px; }
  .logo { border: 1px solid #fff; color: #fff; font-weight: bold; padding: 1px 4px; font-size: 13px; }
  .nav { color: #000; font-size: 13px; }
  .wrap { padding: 10px 16px; max-width: 1100px; }
  .row { display: flex; gap: 6px; margin: 6px 0; font-size: 14px; }
  .rank { color: #828282; width: 26px; text-align: right; }
  .title a { color: #000; text-decoration: none; }
  .dom { color: #828282; font-size: 12px; margin: 2px 0 0 32px; }
  .comment { margin: 10px 0; font-size: 13px; line-height: 1.5; }
  .meta { color: #828282; font-size: 11px; }
  h1 { font-size: 22px; }
  p { font-size: 15px; line-height: 1.6; max-width: 760px; }
</style></head><body>${body}</body></html>`;

const HN_TOPBAR = `<div class="topbar">
  <span class="logo">Y</span>
  <span class="nav"><b>Hacker News</b> &nbsp; new | past | comments | ask | show | jobs</span>
</div>`;

function story(rank, title, host, pts, by, cmts) {
  return `<div class="row"><span class="rank">${rank}.</span>
    <span class="title"><a href="#">${title}</a> <span class="meta">(${host})</span></span></div>
    <div class="dom">${pts} points by ${by} ${rank * 7} minutes ago | hide | ${cmts} comments</div>`;
}

const FRONT = SHELL(
  HN_TOPBAR +
    `<div class="wrap">` +
    story(
      1,
      "Show HN: Memory Inception — KV banks instead of DOM in the prompt",
      "github.com",
      487,
      "ghostdev",
      213,
    ) +
    story(
      2,
      "Llama 3.1 attention internals, annotated",
      "arxiv.org",
      392,
      "kvhead",
      154,
    ) +
    story(
      3,
      "The hidden cost of stuffing the DOM into every agent step",
      "ghostbrowser.dev",
      311,
      "peyton",
      98,
    ) +
    story(
      4,
      "A10G vs H100 for 8B inference: a field report",
      "modal.com",
      256,
      "gpuwrangler",
      76,
    ) +
    story(
      5,
      "Pre-RoPE key/value projection, explained",
      "blog.eleuther.ai",
      204,
      "rotary",
      41,
    ) +
    story(
      6,
      "Playwright loops that don't melt your context window",
      "news.ycombinator.com",
      188,
      "loophero",
      33,
    ) +
    story(
      7,
      "Ask HN: how do you measure real token savings honestly?",
      "news.ycombinator.com",
      142,
      "tokenhonest",
      120,
    ) +
    `</div>`,
);

const ITEM = SHELL(
  HN_TOPBAR +
    `<div class="wrap">
      <div class="row"><span class="rank">1.</span>
        <span class="title"><a href="#">Show HN: Memory Inception — KV banks instead of DOM in the prompt</a></span></div>
      <div class="dom">487 points by ghostdev 41 minutes ago | hide | 213 comments</div>
      <div class="comment"><span class="meta">kvhead 38 min</span><br>This is the cleanest write-up of pre-RoPE injection I've seen. Injecting at L∈{8,12,16,20} is a nice middle ground.</div>
      <div class="comment" style="margin-left:24px"><span class="meta">ghostdev 31 min</span><br>Exactly. We compile one bank per page type, not per exact DOM hash — HN comment counts vary so exact hashes never match.</div>
      <div class="comment"><span class="meta">tokenhonest 22 min</span><br>What are the honest savings? 14k DOM tokens vs ~200 prompt tokens + a 312-slot bank is a real win, but only if it still completes the task.</div>
      <div class="comment" style="margin-left:24px"><span class="meta">peyton 18 min</span><br>It completes. Baseline climbs ~14k/step; MI stays flat near 200. The console shows both lines live.</div>
    </div>`,
);

const ARTICLE = SHELL(
  `<div class="wrap">
     <h1>A10G vs H100 for 8B inference: a field report</h1>
     <p class="meta">modal.com · 9 min read</p>
     <p>For a single Llama-3.1-8B-Instruct replica, the A10G's 24&nbsp;GB is enough headroom to hold the model in bf16 plus a few hundred KV-bank slots without paging.</p>
     <p>The interesting variable isn't raw FLOPs — it's how much prompt you avoid recomputing every step. An agent that re-sends the full DOM each navigation pays an O(layers × T_prompt) tax that dwarfs the model's own cost.</p>
     <p>Replacing that prompt with a precomputed latent bank turns a 14,000-token page into roughly 200 prompt tokens plus a fixed bank, and the per-step cost stops growing with page size.</p>
   </div>`,
  "#ffffff",
);

const POPUP = SHELL(
  HN_TOPBAR +
    `<div class="wrap">` +
    story(
      1,
      "Show HN: Memory Inception — KV banks instead of DOM in the prompt",
      "github.com",
      487,
      "ghostdev",
      213,
    ) +
    story(
      2,
      "Llama 3.1 attention internals, annotated",
      "arxiv.org",
      392,
      "kvhead",
      154,
    ) +
    story(
      3,
      "The hidden cost of stuffing the DOM into every agent step",
      "ghostbrowser.dev",
      311,
      "peyton",
      98,
    ) +
    `</div>
     <div style="position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;">
       <div style="background:#fff;max-width:440px;padding:24px 28px;border-radius:8px;font-family:Verdana;">
         <h2 style="margin:0 0 8px;font-size:18px;">We value your privacy</h2>
         <p style="font-size:13px;color:#333;">We use cookies to enhance your browsing experience and analyze our traffic. By clicking "Accept", you consent to our use of cookies.</p>
         <div style="display:flex;gap:10px;margin-top:16px;justify-content:flex-end;">
           <button style="padding:8px 14px;border:1px solid #ccc;background:#fff;border-radius:5px;">Reject</button>
           <button id="accept-cookies" style="padding:8px 16px;border:0;background:#ff6600;color:#fff;border-radius:5px;font-weight:bold;">Accept</button>
         </div>
       </div>
     </div>`,
);

async function renderFrame(page, html) {
  await page.setContent(html, { waitUntil: "load" });
  const buf = await page.screenshot({ type: "jpeg", quality: 50 });
  return buf.toString("base64");
}

const now = () => new Date().toISOString();

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1280, height: 720 },
  });

  const frames = {
    front: await renderFrame(page, FRONT),
    item: await renderFrame(page, ITEM),
    article: await renderFrame(page, ARTICLE),
    popup: await renderFrame(page, POPUP),
  };
  await browser.close();

  // A realistic mi-mode timeline. baseline_tokens = the DOM a baseline run WOULD
  // have sent; visible_tokens = what MI actually sent. The `unknown` external
  // article is the graceful-fallback moment: no bank, DOM is included, ratio dips.
  const steps = [
    {
      key: "hn:front",
      layers: [8, 12, 16, 20],
      active: true,
      slots: 312,
      base: 14200,
      vis: 212,
      frame: "front",
      action: { action: "goto", url: "https://news.ycombinator.com" },
      log: [
        "info",
        "Bank hn:front injected at layers [8, 12, 16, 20] (312 slots)",
      ],
    },
    {
      key: "hn:item",
      layers: [8, 12, 16, 20],
      active: true,
      slots: 240,
      base: 13800,
      vis: 205,
      frame: "item",
      action: { action: "click", selector: "a.storylink" },
      log: [
        "info",
        "Bank hn:item injected at layers [8, 12, 16, 20] (240 slots)",
      ],
    },
    {
      key: "unknown",
      layers: [],
      active: false,
      slots: 0,
      base: 4300,
      vis: 4300,
      frame: "article",
      action: { action: "click", selector: "span.titleline > a" },
      log: [
        "warn",
        "No bank for unknown; plain-prompt fallback (DOM included this step)",
      ],
    },
    {
      key: "popup:demo",
      layers: [8, 12, 16, 20],
      active: true,
      slots: 96,
      base: 1300,
      vis: 198,
      frame: "popup",
      action: { action: "dismiss_modal", selector: "#accept-cookies" },
      log: [
        "info",
        "Bank popup:demo injected — routing modal dismissal (96 slots)",
      ],
    },
    {
      key: "hn:item",
      layers: [8, 12, 16, 20],
      active: true,
      slots: 240,
      base: 13900,
      vis: 210,
      frame: "item",
      action: {
        action: "extract",
        result: {
          score: 487,
          top_commenters: ["kvhead", "ghostdev", "peyton"],
        },
      },
      log: [
        "info",
        "Bank hn:item injected at layers [8, 12, 16, 20] (240 slots)",
      ],
    },
    {
      key: "hn:item",
      layers: [8, 12, 16, 20],
      active: true,
      slots: 240,
      base: 13900,
      vis: 208,
      frame: "item",
      action: {
        action: "done",
        result: {
          answer:
            "Top story: Show HN: Memory Inception (487 pts, 213 comments)",
        },
      },
      log: ["info", "Task complete — agent returned final answer"],
    },
  ];

  const events = [];
  let cumB = 0;
  let cumV = 0;
  let step = 0;
  for (const s of steps) {
    step += 1;
    cumB += s.base;
    cumV += s.vis;
    const ratio = Math.round((cumB / Math.max(cumV, 1)) * 10) / 10;

    events.push({
      type: "layer_injection",
      ts: now(),
      layers: s.layers,
      active: s.active,
      page_key: s.key,
      num_slots: s.slots,
    });
    events.push({
      type: "viewport_frame",
      ts: now(),
      jpeg_base64: frames[s.frame],
    });
    events.push({ type: "action", ts: now(), step, action: s.action });
    events.push({
      type: "token_metrics",
      ts: now(),
      session_id: "mi-hn-live",
      step,
      mode: "mi",
      visible_tokens: s.vis,
      baseline_tokens: s.base,
      cum_visible: cumV,
      cum_baseline: cumB,
      kv_savings_ratio: ratio,
    });
    events.push({ type: "log", ts: now(), level: s.log[0], message: s.log[1] });
  }

  const out = join(HERE, "real-run.json");
  writeFileSync(out, JSON.stringify(events, null, 2));
  console.log(
    `Wrote ${events.length} events (${steps.length} steps) -> ${out}`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
