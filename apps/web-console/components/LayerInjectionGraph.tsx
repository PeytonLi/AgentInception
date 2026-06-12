"use client";

import { NUM_LAYERS, SELECTED_LAYERS } from "@/lib/events";

interface LayerInjectionGraphProps {
  litLayers: number[];
  injectionActive: boolean;
  numSlots: number;
  activePageKey: string | null;
}

const SELECTED = new Set<number>(SELECTED_LAYERS);

export function LayerInjectionGraph({
  litLayers,
  injectionActive,
  numSlots,
  activePageKey,
}: LayerInjectionGraphProps) {
  const lit = new Set(litLayers);

  return (
    <section className="flex min-h-0 flex-col overflow-hidden border border-edge bg-panel">
      <div className="flex items-center justify-between border-b border-edge px-3 py-1.5">
        <span className="text-[10px] font-bold tracking-[0.25em] text-ink-dim">
          LAYER INJECTION GRAPH
        </span>
        <span className="text-[9px] tracking-widest text-ink-dim">
          {injectionActive ? (
            <span className="text-ghost">
              {numSlots} SLOTS · {activePageKey}
            </span>
          ) : (
            "IDLE"
          )}
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden p-2">
        <div className="grid h-full grid-flow-col grid-rows-8 gap-x-3 gap-y-[3px]">
          {Array.from({ length: NUM_LAYERS }, (_, i) => {
            const isSelected = SELECTED.has(i);
            const isLit = isSelected && lit.has(i);
            return (
              <div
                key={i}
                data-testid={`layer-${i}`}
                data-lit={isLit ? "true" : "false"}
                className="flex items-center gap-2"
              >
                <span className="w-5 text-right text-[9px] tabular-nums text-ink-dim">
                  {i}
                </span>
                <div className="relative h-2 flex-1 overflow-hidden rounded-[1px] bg-panel-2">
                  <div
                    className={`absolute inset-0 origin-left transition-all duration-500 ease-out ${
                      isLit
                        ? "layer-lit bg-ghost"
                        : isSelected
                          ? "bg-ghost-dim/40"
                          : "bg-edge"
                    }`}
                    style={{
                      transform: isLit ? "scaleX(1)" : "scaleX(0.12)",
                    }}
                  />
                </div>
                {isSelected && (
                  <span
                    className={`w-7 text-[8px] font-bold tracking-wider transition-colors ${
                      isLit ? "text-ghost" : "text-ghost-dim"
                    }`}
                  >
                    {isLit ? `+${numSlots}` : "CTRL"}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="border-t border-edge px-3 py-1.5 text-[9px] tracking-widest text-ink-dim">
        CONTROL LAYERS L ∈ {"{"}8, 12, 16, 20{"}"} · {NUM_LAYERS} DECODER LAYERS
      </div>
    </section>
  );
}
