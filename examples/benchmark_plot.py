"""Benchmark visualization utilities using matplotlib + decorator registry."""
from __future__ import annotations

import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from nano_omni.types import PipelineMetrics

PlotFn = Callable[..., None]
_PLOTS: dict[str, PlotFn] = {}


def register_plot(name: str) -> Callable[[PlotFn], PlotFn]:
    def _decorator(fn: PlotFn) -> PlotFn:
        if name in _PLOTS:
            raise ValueError(f"Plot '{name}' already registered")
        _PLOTS[name] = fn
        return fn
    return _decorator


def run_all_plots(
    all_metrics: list[PipelineMetrics],
    labels: list[str],
    prompt_lengths: list[int],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, fn in _PLOTS.items():
        output_path = os.path.join(output_dir, f"{name}.png")
        fn(all_metrics=all_metrics, labels=labels,
           prompt_lengths=prompt_lengths, output_path=output_path)


def _save_or_show(fig, output_path: Optional[str]) -> None:
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"[plot] Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


@register_plot("stage_breakdown")
def plot_stage_breakdown(
    all_metrics: list[PipelineMetrics],
    labels: list[str],
    output_path: Optional[str] = None,
    **_kwargs,
) -> None:
    stage_names = [s.name for s in all_metrics[0].stages]
    num_stages = len(stage_names)
    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bottom = np.zeros(len(labels))

    colors = plt.cm.Set2(np.linspace(0, 1, num_stages))
    for si in range(num_stages):
        vals = [m.stages[si].elapsed_s for m in all_metrics]
        ax.bar(x, vals, width, bottom=bottom, label=stage_names[si], color=colors[si])
        bottom += vals

    ax.set_ylabel("Time (s)")
    ax.set_title("Per-Stage Time Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, output_path)


@register_plot("throughput")
def plot_throughput(
    all_metrics: list[PipelineMetrics],
    labels: list[str],
    output_path: Optional[str] = None,
    **_kwargs,
) -> None:
    stage_names = [s.name for s in all_metrics[0].stages]
    num_stages = len(stage_names)
    x = np.arange(len(labels))
    width = 0.8 / num_stages

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    colors = plt.cm.Set2(np.linspace(0, 1, num_stages))
    for si in range(num_stages):
        vals = [m.stages[si].tokens_per_s for m in all_metrics]
        ax.bar(x + si * width - (num_stages - 1) * width / 2, vals, width,
               label=stage_names[si], color=colors[si])

    ax.set_ylabel("Tokens / s")
    ax.set_title("Per-Stage Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, output_path)


@register_plot("latency_vs_length")
def plot_latency_vs_prompt_length(
    all_metrics: list[PipelineMetrics],
    prompt_lengths: list[int],
    output_path: Optional[str] = None,
    **_kwargs,
) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 5))

    totals = [m.total_s for m in all_metrics]
    ax1.plot(prompt_lengths, totals, "o-", color="tab:blue", label="Total latency")
    ax1.set_xlabel("Prompt length (chars)")
    ax1.set_ylabel("Latency (s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    rtfs = [m.rtf for m in all_metrics]
    if any(r is not None for r in rtfs):
        ax2 = ax1.twinx()
        valid_x = [pl for pl, r in zip(prompt_lengths, rtfs) if r is not None]
        valid_y = [r for r in rtfs if r is not None]
        ax2.plot(valid_x, valid_y, "s--", color="tab:red", label="RTF")
        ax2.set_ylabel("RTF (lower is better)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("Latency vs Prompt Length")
    fig.tight_layout()
    _save_or_show(fig, output_path)
