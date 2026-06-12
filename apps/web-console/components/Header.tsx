"use client";

import { AgentMode } from "@/lib/events";
import { ConnectionStatus } from "@/lib/eventReducer";

interface HeaderProps {
  sessionId: string | null;
  mode: AgentMode | null;
  status: ConnectionStatus;
  step: number;
}

const STATUS_LABEL: Record<ConnectionStatus, string> = {
  connecting: "CONNECTING",
  open: "LIVE",
  closed: "RECONNECTING",
};

const STATUS_COLOR: Record<ConnectionStatus, string> = {
  connecting: "text-amber",
  open: "text-ghost",
  closed: "text-baseline",
};

export function Header({ sessionId, mode, status, step }: HeaderProps) {
  return (
    <header className="flex items-center justify-between border-b border-edge bg-panel px-5 py-3">
      <div className="flex items-center gap-3">
        <div className="grid h-8 w-8 place-items-center rounded-sm border border-ghost-dim text-ghost">
          <span className="text-lg leading-none">◉</span>
        </div>
        <div>
          <h1 className="text-sm font-bold tracking-[0.2em] text-ink">
            GHOSTBROWSER<span className="text-ghost"> OS</span>
          </h1>
          <p className="text-[10px] tracking-wider text-ink-dim">
            MEMORY INCEPTION · LATENT KV INJECTION
          </p>
        </div>
      </div>

      <div className="flex items-center gap-6 text-[11px]">
        <Stat label="SESSION" value={sessionId ?? "—"} />
        <ModeBadge mode={mode} />
        <Stat label="STEP" value={step ? String(step) : "—"} />
        <div className="flex items-center gap-2">
          <span
            className={`h-2 w-2 rounded-full ${
              status === "open"
                ? "bg-ghost"
                : status === "closed"
                  ? "bg-baseline"
                  : "bg-amber"
            } ${status === "open" ? "animate-pulse" : ""}`}
          />
          <span className={`font-bold tracking-widest ${STATUS_COLOR[status]}`}>
            {STATUS_LABEL[status]}
          </span>
        </div>
      </div>
    </header>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-end leading-tight">
      <span className="text-[9px] tracking-widest text-ink-dim">{label}</span>
      <span className="max-w-[160px] truncate font-bold text-ink" title={value}>
        {value}
      </span>
    </div>
  );
}

function ModeBadge({ mode }: { mode: AgentMode | null }) {
  const isMi = mode === "mi";
  const label = mode ? mode.toUpperCase() : "—";
  return (
    <div className="flex flex-col items-end leading-tight">
      <span className="text-[9px] tracking-widest text-ink-dim">MODE</span>
      <span
        className={`rounded-sm px-1.5 py-0.5 text-[10px] font-bold tracking-widest ${
          mode === null
            ? "text-ink-dim"
            : isMi
              ? "bg-ghost-dim/40 text-ghost"
              : "bg-baseline/20 text-baseline"
        }`}
      >
        {label}
      </span>
    </div>
  );
}
