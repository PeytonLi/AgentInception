import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SocketLike, useEventFeed } from "../useEventFeed";

/** Hand-driveable fake socket implementing the SocketLike contract. */
class FakeSocket implements SocketLike {
  static instances: FakeSocket[] = [];
  onopen: ((ev?: unknown) => void) | null = null;
  onclose: ((ev?: unknown) => void) | null = null;
  onerror: ((ev?: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  closed = false;

  constructor(public url: string) {
    FakeSocket.instances.push(this);
  }
  close() {
    this.closed = true;
  }
  // Test helpers
  open() {
    this.onopen?.();
  }
  drop() {
    this.onclose?.();
  }
  emit(data: unknown) {
    this.onmessage?.({ data });
  }
}

describe("useEventFeed reconnect", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeSocket.instances = [];
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const factory = (url: string) => new FakeSocket(url);

  it("opens a socket and reaches the open status", () => {
    const { result } = renderHook(() =>
      useEventFeed("ws://x/ws/events", {
        socketFactory: factory,
        baseBackoffMs: 100,
      }),
    );
    expect(FakeSocket.instances).toHaveLength(1);
    act(() => FakeSocket.instances[0].open());
    expect(result.current.state.status).toBe("open");
  });

  it("retries with backoff after a drop and preserves accumulated state", () => {
    const { result } = renderHook(() =>
      useEventFeed("ws://x/ws/events", {
        socketFactory: factory,
        baseBackoffMs: 100,
        maxBackoffMs: 1000,
      }),
    );

    const first = FakeSocket.instances[0];
    act(() => first.open());

    // Receive an event so there is state to preserve.
    act(() =>
      first.emit(
        JSON.stringify({
          type: "token_metrics",
          session_id: "s1",
          step: 1,
          mode: "mi",
          visible_tokens: 200,
          baseline_tokens: 10000,
          cum_visible: 200,
          cum_baseline: 10000,
          kv_savings_ratio: 50,
        }),
      ),
    );
    expect(result.current.state.cumBaseline).toBe(10000);

    // Socket drops -> status closed, no new socket yet.
    act(() => first.drop());
    expect(result.current.state.status).toBe("closed");
    expect(FakeSocket.instances).toHaveLength(1);

    // Backoff elapses -> a new socket is created (retry).
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(FakeSocket.instances).toHaveLength(2);

    // State survived the reconnect.
    expect(result.current.state.cumBaseline).toBe(10000);

    // New socket opens cleanly.
    act(() => FakeSocket.instances[1].open());
    expect(result.current.state.status).toBe("open");
  });

  it("grows the backoff delay across repeated failures", () => {
    renderHook(() =>
      useEventFeed("ws://x/ws/events", {
        socketFactory: factory,
        baseBackoffMs: 100,
        maxBackoffMs: 10000,
      }),
    );

    // 1st socket drops before opening -> retry after 100ms.
    act(() => FakeSocket.instances[0].drop());
    act(() => vi.advanceTimersByTime(99));
    expect(FakeSocket.instances).toHaveLength(1);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeSocket.instances).toHaveLength(2);

    // 2nd drops -> retry after 200ms.
    act(() => FakeSocket.instances[1].drop());
    act(() => vi.advanceTimersByTime(199));
    expect(FakeSocket.instances).toHaveLength(2);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeSocket.instances).toHaveLength(3);
  });

  it("closes the socket on unmount and stops retrying", () => {
    const { unmount } = renderHook(() =>
      useEventFeed("ws://x/ws/events", {
        socketFactory: factory,
        baseBackoffMs: 100,
      }),
    );
    const sock = FakeSocket.instances[0];
    unmount();
    expect(sock.closed).toBe(true);
    // A drop after unmount must not schedule a retry.
    act(() => sock.drop());
    act(() => vi.advanceTimersByTime(5000));
    expect(FakeSocket.instances).toHaveLength(1);
  });
});
