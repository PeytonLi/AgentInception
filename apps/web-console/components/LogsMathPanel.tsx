"use client";

import katex from "katex";
import { useEffect, useMemo, useRef } from "react";
import { SELECTED_LAYERS } from "@/lib/events";
import { LogEntry } from "@/lib/eventReducer";

interface LogsMathPanelProps {
  logs: LogEntry[];
  kvSavingsRatio: number;
  injectionActive: boolean;
}

const EQUATION = String.raw`K^* = [\,K_{\text{prompt}} \,\Vert\, K_{\text{bank}}\,], \quad V^* = [\,V_{\text{prompt}} \,\Vert\, V_{\text{bank}}\,]`;

const LEVEL_COLOR: Record<string, string> = {
  info: "text-ink",
  warn: "text-amber",
  error: "text-baseline",
};

export function LogsMathPanel({
  logs,
  kvSavingsRatio,
  injectionActive,
}: LogsMathPanelProps) {
  const equationHtml = useMemo(
    () => katex.renderToString(EQUATION, { throwOnError: false }),
    [],
  );

  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs.length]);

  return (
    <section className="flex min-h-0 flex-col overflow-hidden border border-edge bg-panel">
      <div className="flex items-center justify-between border-b border-edge px-3 py-1.5">
        <span className="text-[10px] font-bold tracking-[0.25em] text-ink-dim">
          LOGS &amp; MATH
        </span>
        <span
          data-testid="savings-badge"
          className="rounded-sm bg-ghost-dim/30 px-2 py-0.5 text-[10px] font-bold tracking-widest text-ghost"
        >
          KV RATIO {kvSavingsRatio.toFixed(1)}×
        </span>
      </div>

      <div className="border-b border-edge bg-panel-2 px-3 py-2">
        <div
          className="katex-block text-ink"
          dangerouslySetInnerHTML={{ __html: equationHtml }}
        />
        <div className="mt-1 flex items-center gap-2 text-[9px] tracking-widest text-ink-dim">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              injectionActive ? "bg-ghost" : "bg-edge"
            }`}
          />
          injected at L ∈ {"{"}
          {SELECTED_LAYERS.join(", ")}
          {"}"}
        </div>
      </div>

      <div
        ref={scrollRef}
        className="log-scroll min-h-0 flex-1 overflow-y-auto px-3 py-2 text-[11px] leading-relaxed"
      >
        {logs.length === 0 ? (
          <span className="text-ink-dim">awaiting log events…</span>
        ) : (
          logs.map((log) => (
            <div key={log.id} className="flex gap-2 whitespace-pre-wrap">
              <span className="shrink-0 text-ink-dim">
                {log.level.toUpperCase().padEnd(5).slice(0, 5)}
              </span>
              <span className={LEVEL_COLOR[log.level] ?? "text-ink"}>
                {log.message}
              </span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
