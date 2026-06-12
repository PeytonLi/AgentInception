"""Cumulative token metrics for the comparison chart (A3 brief task 5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    cum_visible: int = 0
    cum_baseline: int = 0
    steps: int = 0

    def record(self, visible_tokens: int, baseline_tokens: int) -> None:
        self.cum_visible += int(visible_tokens)
        self.cum_baseline += int(baseline_tokens)
        self.steps += 1

    @property
    def kv_savings_ratio(self) -> float:
        if self.cum_visible <= 0:
            return 0.0
        return round(self.cum_baseline / self.cum_visible, 1)
