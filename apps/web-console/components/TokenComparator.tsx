"use client";

import { useEffect, useRef, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MetricPoint } from "@/lib/eventReducer";

interface TokenComparatorProps {
  history: MetricPoint[];
  cumVisible: number;
  cumBaseline: number;
  kvSavingsRatio: number;
}

function fmt(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

/** Eases a displayed number toward a target for a satisfying ticker effect. */
function useAnimatedNumber(target: number, ms = 600): number {
  const [value, setValue] = useState(target);
  const fromRef = useRef(target);
  const startRef = useRef(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    fromRef.current = value;
    startRef.current = performance.now();
    const from = fromRef.current;
    const tick = (now: number) => {
      const t = Math.min(1, (now - startRef.current) / ms);
      const eased = 1 - (1 - t) ** 3;
      setValue(from + (target - from) * eased);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, ms]);

  return value;
}

export function TokenComparator({
  history,
  cumVisible,
  cumBaseline,
  kvSavingsRatio,
}: TokenComparatorProps) {
  const animatedRatio = useAnimatedNumber(kvSavingsRatio);

  const data =
    history.length > 0
      ? history.map((p) => ({
          step: p.step,
          "Standard prompting": p.cumBaseline,
          "AgentInception MI": p.cumVisible,
        }))
      : [];

  return (
    <section className="flex min-h-0 flex-col overflow-hidden border border-edge bg-panel">
      <div className="flex items-center justify-between border-b border-edge px-3 py-1.5">
        <span className="text-[10px] font-bold tracking-[0.25em] text-ink-dim">
          TOKEN COST COMPARATOR
        </span>
        <div className="flex items-baseline gap-1.5">
          <span
            data-testid="kv-savings"
            className="text-2xl font-bold leading-none text-ghost tabular-nums"
          >
            {animatedRatio.toFixed(1)}×
          </span>
          <span className="text-[10px] tracking-widest text-ink-dim">
            SAVED
          </span>
        </div>
      </div>

      <div className="flex gap-4 px-3 pt-2 text-[11px]">
        <Metric
          label="STANDARD"
          value={fmt(cumBaseline)}
          color="var(--color-baseline)"
        />
        <Metric
          label="GHOST MI"
          value={fmt(cumVisible)}
          color="var(--color-ghost)"
        />
      </div>

      <div className="min-h-0 flex-1 p-1">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 8, right: 12, bottom: 4, left: 0 }}
          >
            <defs>
              <linearGradient id="baselineFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ff5470" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#ff5470" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#1c2533" strokeDasharray="3 3" />
            <XAxis
              dataKey="step"
              stroke="#8b98a9"
              tick={{ fontSize: 11 }}
              label={{
                value: "step",
                position: "insideBottomRight",
                fontSize: 9,
                fill: "#8b98a9",
              }}
            />
            <YAxis
              stroke="#8b98a9"
              tick={{ fontSize: 11 }}
              tickFormatter={fmt}
              width={46}
            />
            <Tooltip
              contentStyle={{
                background: "#0d1118",
                border: "1px solid #1c2533",
                fontSize: 11,
              }}
              labelStyle={{ color: "#8b98a9" }}
              formatter={(v: number) => v.toLocaleString()}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Area
              type="monotone"
              dataKey="Standard prompting"
              stroke="#ff5470"
              strokeWidth={2}
              fill="url(#baselineFill)"
              dot={false}
              isAnimationActive
            />
            <Line
              type="monotone"
              dataKey="AgentInception MI"
              stroke="#38e8c8"
              strokeWidth={3}
              dot={false}
              isAnimationActive
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-sm" style={{ background: color }} />
      <span className="tracking-widest text-ink-dim">{label}</span>
      <span className="font-bold" style={{ color }}>
        {value}
      </span>
    </div>
  );
}
