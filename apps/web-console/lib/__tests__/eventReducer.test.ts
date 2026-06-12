import { describe, expect, it } from "vitest";
import {
  DashboardState,
  eventReducer,
  initialState,
} from "../eventReducer";
import { GhostEvent } from "../events";

function apply(state: DashboardState, event: GhostEvent): DashboardState {
  return eventReducer(state, { kind: "event", event });
}

describe("eventReducer", () => {
  it("stores the latest viewport frame", () => {
    const s = apply(initialState, {
      type: "viewport_frame",
      jpeg_base64: "abc123",
    });
    expect(s.latestFrame).toBe("abc123");
    expect(s.eventCount).toBe(1);
  });

  it("lights up the listed layers on an active injection", () => {
    const s = apply(initialState, {
      type: "layer_injection",
      layers: [8, 12, 16, 20],
      active: true,
      page_key: "hn:front",
      num_slots: 312,
    });
    expect(s.injectionActive).toBe(true);
    expect(s.litLayers).toEqual([8, 12, 16, 20]);
    expect(s.activePageKey).toBe("hn:front");
    expect(s.numSlots).toBe(312);
    expect(s.popupFlashSeq).toBe(0);
  });

  it("clears the graph on an inactive injection", () => {
    let s = apply(initialState, {
      type: "layer_injection",
      layers: [8, 12, 16, 20],
      active: true,
      page_key: "hn:item",
      num_slots: 200,
    });
    s = apply(s, {
      type: "layer_injection",
      layers: [],
      active: false,
      page_key: "unknown",
      num_slots: 0,
    });
    expect(s.injectionActive).toBe(false);
    expect(s.litLayers).toEqual([]);
    expect(s.activePageKey).toBeNull();
    expect(s.numSlots).toBe(0);
  });

  it("bumps popupFlashSeq only for an active popup:demo injection", () => {
    let s = apply(initialState, {
      type: "layer_injection",
      layers: [8, 12, 16, 20],
      active: true,
      page_key: "popup:demo",
      num_slots: 96,
    });
    expect(s.popupFlashSeq).toBe(1);
    // Non-popup injection must not bump it.
    s = apply(s, {
      type: "layer_injection",
      layers: [8, 12, 16, 20],
      active: true,
      page_key: "hn:front",
      num_slots: 312,
    });
    expect(s.popupFlashSeq).toBe(1);
    // An inactive popup event must not bump it either.
    s = apply(s, {
      type: "layer_injection",
      layers: [],
      active: false,
      page_key: "popup:demo",
      num_slots: 0,
    });
    expect(s.popupFlashSeq).toBe(1);
  });

  it("tracks cumulative token metrics and appends history", () => {
    let s = apply(initialState, {
      type: "token_metrics",
      session_id: "sess-1",
      step: 1,
      mode: "mi",
      visible_tokens: 212,
      baseline_tokens: 14200,
      cum_visible: 212,
      cum_baseline: 14200,
      kv_savings_ratio: 67.0,
    });
    expect(s.sessionId).toBe("sess-1");
    expect(s.mode).toBe("mi");
    expect(s.cumVisible).toBe(212);
    expect(s.cumBaseline).toBe(14200);
    expect(s.kvSavingsRatio).toBe(67.0);
    expect(s.lastStep).toBe(1);
    expect(s.metricHistory).toHaveLength(1);

    s = apply(s, {
      type: "token_metrics",
      session_id: "sess-1",
      step: 2,
      mode: "mi",
      visible_tokens: 220,
      baseline_tokens: 16000,
      cum_visible: 432,
      cum_baseline: 30200,
      kv_savings_ratio: 69.9,
    });
    expect(s.cumVisible).toBe(432);
    expect(s.cumBaseline).toBe(30200);
    expect(s.metricHistory).toHaveLength(2);
    expect(s.metricHistory[1]).toEqual({
      step: 2,
      cumBaseline: 30200,
      cumVisible: 432,
    });
  });

  it("appends actions and logs in order with stable ids", () => {
    let s = apply(initialState, {
      type: "action",
      step: 1,
      action: { action: "goto", url: "https://news.ycombinator.com" },
    });
    s = apply(s, {
      type: "log",
      level: "info",
      message: "Bank hn:front injected at layers [8, 12, 16, 20]",
    });
    s = apply(s, { type: "log", level: "warn", message: "fallback" });
    expect(s.actions).toHaveLength(1);
    expect(s.actions[0].action.action).toBe("goto");
    expect(s.logs).toHaveLength(2);
    expect(s.logs[0].id).toBe(0);
    expect(s.logs[1].id).toBe(1);
    expect(s.logs[1].level).toBe("warn");
  });

  it("ignores unknown event types without mutating state", () => {
    const before = apply(initialState, {
      type: "viewport_frame",
      jpeg_base64: "frame",
    });
    const after = eventReducer(before, {
      kind: "event",
      event: { type: "totally_unknown", foo: 1 } as unknown as GhostEvent,
    });
    expect(after).toEqual(before);
    expect(after.eventCount).toBe(before.eventCount);
  });

  it("preserves accumulated state across a status change (reconnect)", () => {
    const populated = apply(initialState, {
      type: "token_metrics",
      session_id: "sess-1",
      step: 5,
      mode: "mi",
      visible_tokens: 200,
      baseline_tokens: 12000,
      cum_visible: 1000,
      cum_baseline: 60000,
      kv_savings_ratio: 60,
    });
    const closed = eventReducer(populated, {
      kind: "status",
      status: "closed",
    });
    expect(closed.status).toBe("closed");
    expect(closed.cumVisible).toBe(1000);
    expect(closed.cumBaseline).toBe(60000);
    expect(closed.lastStep).toBe(5);

    const reopened = eventReducer(closed, { kind: "status", status: "open" });
    expect(reopened.status).toBe("open");
    expect(reopened.cumBaseline).toBe(60000);
  });
});
