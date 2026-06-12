// Event schema mirrors CONTRACTS.md §7 (`/ws/events`).
// The web-console is the only consumer of this feed.

export const SELECTED_LAYERS = [8, 12, 16, 20] as const;
export const NUM_LAYERS = 32;
export const POPUP_PAGE_KEY = "popup:demo";

export type AgentMode = "baseline" | "mi";

export interface LayerInjectionEvent {
  type: "layer_injection";
  layers: number[];
  active: boolean;
  page_key: string;
  num_slots: number;
  ts?: string;
}

export interface TokenMetricsEvent {
  type: "token_metrics";
  session_id: string;
  step: number;
  mode: AgentMode;
  visible_tokens: number;
  baseline_tokens: number;
  cum_visible: number;
  cum_baseline: number;
  kv_savings_ratio: number;
  ts?: string;
}

export interface ActionEvent {
  type: "action";
  step: number;
  action: Record<string, unknown>;
  ts?: string;
}

export interface ViewportFrameEvent {
  type: "viewport_frame";
  jpeg_base64: string;
  ts?: string;
}

export interface LogEvent {
  type: "log";
  level: "info" | "warn" | "error" | string;
  message: string;
  ts?: string;
}

export type GhostEvent =
  | LayerInjectionEvent
  | TokenMetricsEvent
  | ActionEvent
  | ViewportFrameEvent
  | LogEvent;

/** Narrowing parse for an unknown WS payload. Returns null on anything malformed. */
export function parseEvent(raw: unknown): GhostEvent | null {
  if (typeof raw !== "object" || raw === null) return null;
  const t = (raw as { type?: unknown }).type;
  if (typeof t !== "string") return null;
  switch (t) {
    case "layer_injection":
    case "token_metrics":
    case "action":
    case "viewport_frame":
    case "log":
      return raw as GhostEvent;
    default:
      return null;
  }
}
