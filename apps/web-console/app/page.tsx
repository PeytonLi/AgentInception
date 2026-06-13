"use client";

import { Header } from "@/components/Header";
import { LayerInjectionGraph } from "@/components/LayerInjectionGraph";
import { LogsMathPanel } from "@/components/LogsMathPanel";
import { StepTimeline } from "@/components/StepTimeline";
import { TokenComparator } from "@/components/TokenComparator";
import { ViewportPanel } from "@/components/ViewportPanel";
import { useEventFeed } from "@/lib/useEventFeed";

const WS_URL =
  process.env.NEXT_PUBLIC_INFERENCE_WS ?? "ws://localhost:8000/ws/events";

export default function Dashboard() {
  const { state } = useEventFeed(WS_URL);

  return (
    <main className="bg-grid relative flex h-screen flex-col">
      <Header
        sessionId={state.sessionId}
        mode={state.mode}
        status={state.status}
        step={state.lastStep}
      />

      {state.status === "closed" && (
        <div
          data-testid="reconnect-banner"
          className="flex items-center justify-center gap-2 border-b border-baseline/40 bg-baseline/10 px-4 py-1.5 text-[11px] font-bold tracking-widest text-baseline"
        >
          <span className="h-2 w-2 animate-pulse rounded-full bg-baseline" />
          ENGINE LINK LOST — RECONNECTING… LAST FRAME HELD
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col gap-2 p-2">
        <div className="grid h-[55vh] flex-shrink-0 grid-cols-2 grid-rows-2 gap-2">
          <ViewportPanel
            frame={state.latestFrame}
            popupFlashSeq={state.popupFlashSeq}
            activePageKey={state.activePageKey}
          />
          <TokenComparator
            history={state.metricHistory}
            cumVisible={state.cumVisible}
            cumBaseline={state.cumBaseline}
            kvSavingsRatio={state.kvSavingsRatio}
          />
          <LayerInjectionGraph
            litLayers={state.litLayers}
            injectionActive={state.injectionActive}
            numSlots={state.numSlots}
            activePageKey={state.activePageKey}
          />
          <LogsMathPanel
            logs={state.logs}
            kvSavingsRatio={state.kvSavingsRatio}
            injectionActive={state.injectionActive}
            numSlots={state.numSlots}
            domTokenCount={state.domTokenCount}
          />
        </div>
        <StepTimeline
          actions={state.actions}
          metricHistory={state.metricHistory}
          bankUsed={state.injectionActive}
          activePageKey={state.activePageKey}
        />
      </div>
    </main>
  );
}
