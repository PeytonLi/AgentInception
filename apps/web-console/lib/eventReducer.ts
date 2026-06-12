// Pure, unit-testable store for the live event feed.
// All panels read from this single state; the WS hook is the only dispatcher.

import { AgentMode, GhostEvent, POPUP_PAGE_KEY } from "./events";

export type ConnectionStatus = "connecting" | "open" | "closed";

export interface LogEntry {
  id: number;
  level: string;
  message: string;
  ts: string;
}

export interface ActionEntry {
  step: number;
  action: Record<string, unknown>;
}

export interface MetricPoint {
  step: number;
  cumBaseline: number;
  cumVisible: number;
}

export interface DashboardState {
  sessionId: string | null;
  mode: AgentMode | null;

  // Viewport
  latestFrame: string | null;

  // Layer injection graph
  injectionActive: boolean;
  litLayers: number[];
  activePageKey: string | null;
  numSlots: number;
  /** Monotonic counter; bumped whenever a popup:demo bank is injected. */
  popupFlashSeq: number;

  // Token comparator
  cumVisible: number;
  cumBaseline: number;
  kvSavingsRatio: number;
  lastStep: number;
  metricHistory: MetricPoint[];

  // Logs & actions
  logs: LogEntry[];
  actions: ActionEntry[];

  // Connection
  status: ConnectionStatus;

  // Bookkeeping
  eventCount: number;
  logSeq: number;
}

export const initialState: DashboardState = {
  sessionId: null,
  mode: null,
  latestFrame: null,
  injectionActive: false,
  litLayers: [],
  activePageKey: null,
  numSlots: 0,
  popupFlashSeq: 0,
  cumVisible: 0,
  cumBaseline: 0,
  kvSavingsRatio: 0,
  lastStep: 0,
  metricHistory: [],
  logs: [],
  actions: [],
  status: "connecting",
  eventCount: 0,
  logSeq: 0,
};

export type ReducerAction =
  | { kind: "event"; event: GhostEvent }
  | { kind: "status"; status: ConnectionStatus }
  | { kind: "reset" };

const MAX_LOGS = 500;
const MAX_HISTORY = 400;

export function eventReducer(
  state: DashboardState,
  action: ReducerAction,
): DashboardState {
  switch (action.kind) {
    case "reset":
      return { ...initialState, status: state.status };

    case "status":
      // Connection transitions never wipe accumulated data (reconnect-safe).
      return { ...state, status: action.status };

    case "event":
      return applyEvent(state, action.event);

    default:
      return state;
  }
}

function applyEvent(state: DashboardState, event: GhostEvent): DashboardState {
  const base = { ...state, eventCount: state.eventCount + 1 };

  switch (event.type) {
    case "viewport_frame":
      return { ...base, latestFrame: event.jpeg_base64 };

    case "layer_injection": {
      const isPopup = event.active && event.page_key === POPUP_PAGE_KEY;
      return {
        ...base,
        injectionActive: event.active,
        litLayers: event.active ? [...event.layers] : [],
        activePageKey: event.active ? event.page_key : null,
        numSlots: event.active ? event.num_slots : 0,
        popupFlashSeq: isPopup ? base.popupFlashSeq + 1 : base.popupFlashSeq,
      };
    }

    case "token_metrics": {
      const point: MetricPoint = {
        step: event.step,
        cumBaseline: event.cum_baseline,
        cumVisible: event.cum_visible,
      };
      const history = [...base.metricHistory, point].slice(-MAX_HISTORY);
      return {
        ...base,
        sessionId: event.session_id,
        mode: event.mode,
        cumVisible: event.cum_visible,
        cumBaseline: event.cum_baseline,
        kvSavingsRatio: event.kv_savings_ratio,
        lastStep: event.step,
        metricHistory: history,
      };
    }

    case "action": {
      const entry: ActionEntry = { step: event.step, action: event.action };
      return { ...base, actions: [...base.actions, entry].slice(-MAX_HISTORY) };
    }

    case "log": {
      const entry: LogEntry = {
        id: base.logSeq,
        level: event.level,
        message: event.message,
        ts: event.ts ?? new Date(0).toISOString(),
      };
      return {
        ...base,
        logSeq: base.logSeq + 1,
        logs: [...base.logs, entry].slice(-MAX_LOGS),
      };
    }

    default:
      // Unknown event types are ignored entirely (no state change).
      return state;
  }
}
