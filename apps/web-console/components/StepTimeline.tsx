"use client";

import { useEffect, useRef } from "react";
import { ActionEntry, MetricPoint } from "@/lib/eventReducer";

interface StepTimelineProps {
  actions: ActionEntry[];
  metricHistory: MetricPoint[];
  bankUsed: boolean;
  activePageKey: string | null;
}

function fmt(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

const ACTION_LABEL: Record<string, string> = {
  goto: "NAVIGATE",
  click: "CLICK",
  dismiss_modal: "DISMISS",
  extract: "EXTRACT",
  done: "DONE",
};

const ACTION_COLOR: Record<string, string> = {
  goto: "text-cyan",
  click: "text-ghost",
  dismiss_modal: "text-amber",
  extract: "text-violet",
  done: "text-green",
};

export function StepTimeline({
  actions, metricHistory, bankUsed, activePageKey,
}: StepTimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [actions.length]);
  const metricsByStep = new Map<number, MetricPoint>();
  for (const m of metricHistory) metricsByStep.set(m.step, m);
  return (
    <section className="flex min-h-0 flex-col overflow-hidden border border-edge bg-panel">
      <div className="flex items-center justify-between border-b border-edge px-3 py-1.5">
        <span className="text-[10px] font-bold tracking-[0.25em] text-ink-dim">AGENT THOUGHT PROCESS</span>
        <span className="text-[9px] tracking-widest text-ink-dim">
          {bankUsed ? <span className="text-ghost">BANK ACTIVE &middot; {activePageKey}</span> : <span className="text-ink-dim">NO BANK</span>}
        </span>
      </div>
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
        {actions.length === 0 ? (
          <div className="flex items-center gap-2 text-xs text-ink-dim">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ghost" />
            awaiting agent actions...
          </div>
        ) : (
          <div className="space-y-2.5">
            {actions.map((entry, i) => {
              const action = entry.action;
              const type = String(action.action ?? "?");
              const thought = String(action.thought ?? "");
              const metric = metricsByStep.get(entry.step);
              return (
                <div key={i} className="relative rounded-sm border border-edge/50 bg-panel-2 px-2.5 py-2">
                  <div className="mb-1.5 flex items-center gap-2">
                    <span className="rounded-sm bg-edge/60 px-1.5 py-0.5 text-[10px] font-bold tabular-nums text-ink-dim">STEP {entry.step}</span>
                    <span className={"text-[11px] font-bold " + (ACTION_COLOR[type] ?? "text-ink")}>{ACTION_LABEL[type] ?? type.toUpperCase()}</span>
                    {metric && <span className="ml-auto text-[10px] tabular-nums text-ghost">{fmt(metric.cumVisible)} tokens{metric.cumBaseline > metric.cumVisible && <span className="ml-1 text-baseline">({fmt(metric.cumBaseline)} w/o bank)</span>}</span>}
                  </div>
                  {thought && <div className="rounded-sm border-l-2 border-ghost-dim/60 bg-panel-2/80 px-2 py-1.5 text-[11px] leading-relaxed text-ink-dim italic">&ldquo;{thought}&rdquo;</div>}
                  <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-ink-dim">
                    {action.url && <span>url: <span className="text-ink">{String(action.url)}</span></span>}
                    {action.selector && <span>selector: <span className="text-ink">{String(action.selector)}</span></span>}
                    {action.result && <span>result: <span className="text-ink">{JSON.stringify(action.result).slice(0, 80)}</span></span>}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
