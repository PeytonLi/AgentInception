"use client";

import { useEffect, useReducer, useRef } from "react";
import { parseEvent } from "./events";
import {
  DashboardState,
  ReducerAction,
  eventReducer,
  initialState,
} from "./eventReducer";

/** Minimal structural type so tests can inject a fake socket. */
export interface SocketLike {
  onopen: ((ev?: unknown) => void) | null;
  onclose: ((ev?: unknown) => void) | null;
  onerror: ((ev?: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  close(): void;
}

export type SocketFactory = (url: string) => SocketLike;

export interface UseEventFeedOptions {
  /** Injectable for tests; defaults to the global WebSocket. */
  socketFactory?: SocketFactory;
  /** First retry delay in ms. */
  baseBackoffMs?: number;
  /** Backoff ceiling in ms. */
  maxBackoffMs?: number;
  /** Disable the connection entirely (SSR / opt-out). */
  enabled?: boolean;
}

export interface UseEventFeedResult {
  state: DashboardState;
  dispatch: React.Dispatch<ReducerAction>;
}

const DEFAULT_BASE_BACKOFF = 500;
const DEFAULT_MAX_BACKOFF = 10_000;

function defaultFactory(url: string): SocketLike {
  return new WebSocket(url) as unknown as SocketLike;
}

/**
 * Subscribes to the inference engine `/ws/events` feed, parses each payload,
 * and dispatches typed events into the pure `eventReducer`.
 *
 * Reconnects with exponential backoff. Accumulated state is preserved across
 * reconnects — only the connection status flips.
 */
export function useEventFeed(
  url: string,
  options: UseEventFeedOptions = {},
): UseEventFeedResult {
  const [state, dispatch] = useReducer(eventReducer, initialState);

  const baseBackoff = options.baseBackoffMs ?? DEFAULT_BASE_BACKOFF;
  const maxBackoff = options.maxBackoffMs ?? DEFAULT_MAX_BACKOFF;
  const enabled = options.enabled ?? true;
  const factory = options.socketFactory ?? defaultFactory;

  // Keep the latest callbacks/values in refs so the connect loop is stable.
  const factoryRef = useRef(factory);
  factoryRef.current = factory;

  useEffect(() => {
    if (!enabled || !url) return;

    let closedByUs = false;
    let attempt = 0;
    let socket: SocketLike | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      dispatch({ kind: "status", status: "connecting" });
      let sock: SocketLike;
      try {
        sock = factoryRef.current(url);
      } catch {
        scheduleRetry();
        return;
      }
      socket = sock;

      sock.onopen = () => {
        attempt = 0;
        dispatch({ kind: "status", status: "open" });
      };

      sock.onmessage = (ev: { data: unknown }) => {
        let payload: unknown = ev.data;
        if (typeof payload === "string") {
          try {
            payload = JSON.parse(payload);
          } catch {
            return;
          }
        }
        const event = parseEvent(payload);
        if (event) dispatch({ kind: "event", event });
      };

      sock.onerror = () => {
        // Surface as a closed connection; onclose drives the retry.
        try {
          sock.close();
        } catch {
          /* ignore */
        }
      };

      sock.onclose = () => {
        if (closedByUs) return;
        dispatch({ kind: "status", status: "closed" });
        scheduleRetry();
      };
    };

    const scheduleRetry = () => {
      const delay = Math.min(baseBackoff * 2 ** attempt, maxBackoff);
      attempt += 1;
      retryTimer = setTimeout(connect, delay);
    };

    connect();

    return () => {
      closedByUs = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (socket) {
        try {
          socket.close();
        } catch {
          /* ignore */
        }
      }
    };
  }, [url, enabled, baseBackoff, maxBackoff]);

  return { state, dispatch };
}
