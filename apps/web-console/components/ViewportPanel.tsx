"use client";

import { useEffect, useRef, useState } from "react";

interface ViewportPanelProps {
  frame: string | null;
  popupFlashSeq: number;
  activePageKey: string | null;
}

export function ViewportPanel({
  frame,
  popupFlashSeq,
  activePageKey,
}: ViewportPanelProps) {
  const [flashing, setFlashing] = useState(false);
  const flashKey = useRef(0);
  const seenSeq = useRef(0);

  // Real frames arrive ~300 ms (and can burst on reconnect). Decouple the
  // painted frame from the prop with a single rAF: only the newest frame is
  // ever committed to the <img>, so stale frames are dropped and we never
  // queue work faster than the display can paint it (no lag/leak over long runs).
  const [paintedFrame, setPaintedFrame] = useState<string | null>(frame);
  const pendingFrame = useRef<string | null>(frame);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    pendingFrame.current = frame;
    if (rafRef.current !== null) return; // a paint is already scheduled
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      setPaintedFrame((prev) =>
        prev === pendingFrame.current ? prev : pendingFrame.current,
      );
    });
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [frame]);

  useEffect(() => {
    if (popupFlashSeq > seenSeq.current) {
      seenSeq.current = popupFlashSeq;
      flashKey.current += 1;
      setFlashing(true);
      const t = setTimeout(() => setFlashing(false), 1500);
      return () => clearTimeout(t);
    }
  }, [popupFlashSeq]);

  return (
    <Panel title="LIVE VIEWPORT MIRROR" accent={activePageKey}>
      <div className="relative h-full w-full overflow-hidden rounded-sm bg-black">
        {paintedFrame ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={`data:image/jpeg;base64,${paintedFrame}`}
            alt="agent viewport"
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-xs text-ink-dim">
            <span className="animate-pulse">awaiting viewport stream…</span>
          </div>
        )}

        {flashing && (
          <div
            key={flashKey.current}
            data-testid="popup-flash"
            className="crimson-flash pointer-events-none absolute inset-0 z-10"
          />
        )}

        {flashing && (
          <div className="pointer-events-none absolute left-3 top-3 z-20 rounded-sm bg-crimson/90 px-2 py-1 text-[10px] font-bold tracking-widest text-white">
            POPUP BANK ROUTED
          </div>
        )}
      </div>
    </Panel>
  );
}

function Panel({
  title,
  accent,
  children,
}: {
  title: string;
  accent: string | null;
  children: React.ReactNode;
}) {
  return (
    <section className="flex min-h-0 flex-col overflow-hidden border border-edge bg-panel">
      <PanelHeader title={title} right={accent} />
      <div className="min-h-0 flex-1 p-2">{children}</div>
    </section>
  );
}

function PanelHeader({
  title,
  right,
}: {
  title: string;
  right: string | null;
}) {
  return (
    <div className="flex items-center justify-between border-b border-edge px-3 py-1.5">
      <span className="text-[10px] font-bold tracking-[0.25em] text-ink-dim">
        {title}
      </span>
      {right && (
        <span className="rounded-sm bg-panel-2 px-1.5 py-0.5 text-[9px] font-bold tracking-widest text-ghost">
          {right}
        </span>
      )}
    </div>
  );
}
