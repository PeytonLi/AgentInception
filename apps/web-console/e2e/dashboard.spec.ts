import { expect, test } from "@playwright/test";

/**
 * Self-contained smoke test: we replace window.WebSocket with a fake that
 * replays a canned CONTRACTS §7 sequence, so the test needs no backend.
 * Verifies all 4 panels render and layer bars light up within 2s of a
 * `layer_injection` event.
 */
const INIT_SCRIPT = `
  const RealWS = window.WebSocket;
  class FakeWS {
    constructor(url) {
      this.url = url;
      this.onopen = null; this.onclose = null;
      this.onerror = null; this.onmessage = null;
      setTimeout(() => {
        if (this.onopen) this.onopen();
        this._send({ type: "layer_injection", layers: [8,12,16,20], active: true, page_key: "hn:front", num_slots: 312 });
        this._send({ type: "viewport_frame", jpeg_base64: "/9j/4AAQSkZJRg==" });
        this._send({ type: "action", step: 1, action: { action: "goto", url: "https://news.ycombinator.com" } });
        this._send({ type: "token_metrics", session_id: "e2e", step: 1, mode: "mi", visible_tokens: 210, baseline_tokens: 1600, cum_visible: 210, cum_baseline: 1600, kv_savings_ratio: 7.6 });
        this._send({ type: "log", level: "info", message: "Bank hn:front injected at layers [8, 12, 16, 20]" });
      }, 50);
    }
    _send(obj) { if (this.onmessage) this.onmessage({ data: JSON.stringify(obj) }); }
    close() {}
  }
  // Only fake the inference feed; let Next.js HMR and others use the real socket.
  window.WebSocket = function (url, protocols) {
    if (String(url).includes("/ws/events")) return new FakeWS(url);
    return new RealWS(url, protocols);
  };
`;

test("renders 4 panels and lights up control layers from a layer_injection", async ({
  page,
}) => {
  await page.addInitScript(INIT_SCRIPT);
  await page.goto("/");

  await expect(page.getByText("LIVE VIEWPORT MIRROR")).toBeVisible();
  await expect(page.getByText("TOKEN COST COMPARATOR")).toBeVisible();
  await expect(page.getByText("LAYER INJECTION GRAPH")).toBeVisible();
  await expect(page.getByText("LOGS & MATH")).toBeVisible();

  // Control layers light up within 2s of the injection event.
  for (const layer of [8, 12, 16, 20]) {
    await expect(page.getByTestId(`layer-${layer}`)).toHaveAttribute(
      "data-lit",
      "true",
      { timeout: 2000 },
    );
  }

  // Session + mode propagate to the header.
  await expect(page.getByText("e2e")).toBeVisible();

  // Savings badge reflects the metric.
  await expect(page.getByTestId("savings-badge")).toContainText("7.6×");
});
