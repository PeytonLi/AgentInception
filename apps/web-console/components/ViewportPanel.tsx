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
        {frame ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={`data:image/jpeg;base64,${frame}`}
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

function PanelHeader({ title, right }: { title: string; right: string | null }) {
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
