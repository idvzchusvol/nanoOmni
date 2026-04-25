#!/usr/bin/env python3
"""
nanoOmni benchmark: measure per-stage latency and throughput.

Usage:
    python examples/benchmark.py --config configs/qwen25_omni.yaml
    python examples/benchmark.py --config configs/qwen3_omni.yaml --num-runs 3 --output-dir bench_results
"""
from __future__ import annotations

import argparse
import os
import time

from nano_omni.models import build_omni_pipeline
from nano_omni.models.qwen_omni import load_model_config
from nano_omni.types import OmniRequest, PipelineMetrics

PROMPTS = [
    "你好",
    "请介绍一下你自己。",
    "请用简短的语言解释一下什么是人工智能，以及它在日常生活中的应用。",
    "请详细介绍一下深度学习的基本原理，包括神经网络的结构、反向传播算法的工作方式，以及常见的优化方法。",
]


def run_benchmark(pipeline, prompts: list[str], num_runs: int) -> list[PipelineMetrics]:
    all_metrics: list[PipelineMetrics] = []
    for i, text in enumerate(prompts):
        run_metrics: list[PipelineMetrics] = []
        for r in range(num_runs):
            req = OmniRequest(request_id=f"bench_{i}_run{r}", text=text)
            _, metrics = pipeline.run(requests=[req])
            run_metrics.append(metrics)

        avg = _average_metrics(run_metrics)
        all_metrics.append(avg)
    return all_metrics


def _average_metrics(runs: list[PipelineMetrics]) -> PipelineMetrics:
    from nano_omni.types import StageMetrics

    num = len(runs)
    num_stages = len(runs[0].stages)
    avg_stages = []
    for si in range(num_stages):
        avg_stages.append(StageMetrics(
            name=runs[0].stages[si].name,
            elapsed_s=sum(r.stages[si].elapsed_s for r in runs) / num,
            num_tokens=sum(r.stages[si].num_tokens for r in runs) // num,
            ttft_s=(sum(r.stages[si].ttft_s for r in runs if r.stages[si].ttft_s is not None) / num
                    if runs[0].stages[si].ttft_s is not None else None),
        ))

    audio_durations = [r.audio_duration_s for r in runs if r.audio_duration_s is not None]
    avg_audio = sum(audio_durations) / len(audio_durations) if audio_durations else None

    return PipelineMetrics(
        stages=avg_stages,
        total_s=sum(r.total_s for r in runs) / num,
        audio_duration_s=avg_audio,
    )


def print_table(all_metrics: list[PipelineMetrics], prompts: list[str]) -> None:
    print("\n" + "=" * 90)
    print(f"{'Prompt':<30} | {'Stage':<15} | {'Time(s)':>8} | {'Tokens':>7} | {'Tok/s':>8}")
    print("-" * 90)

    for i, (m, text) in enumerate(zip(all_metrics, prompts)):
        label = text[:27] + "..." if len(text) > 30 else text
        for si, sm in enumerate(m.stages):
            prefix = label if si == 0 else ""
            print(f"{prefix:<30} | {sm.name:<15} | {sm.elapsed_s:>8.3f} | {sm.num_tokens:>7} | {sm.tokens_per_s:>8.1f}")
        rtf_str = f"{m.rtf:.3f}" if m.rtf is not None else "N/A"
        print(f"{'':30} | {'TOTAL':<15} | {m.total_s:>8.3f} | {'':>7} | RTF={rtf_str}")
        print("-" * 90)

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="nanoOmni benchmark")
    parser.add_argument("--config", default="configs/qwen25_omni.yaml")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    print(f"[bench] Loading model: {cfg.model_path}")
    t0 = time.perf_counter()
    pipeline = build_omni_pipeline(cfg, device=args.device)
    load_time = time.perf_counter() - t0
    print(f"[bench] Model loaded in {load_time:.1f}s")

    print(f"[bench] Running {len(PROMPTS)} prompts x {args.num_runs} runs ...")
    all_metrics = run_benchmark(pipeline, PROMPTS, args.num_runs)
    print_table(all_metrics, PROMPTS)

    if args.output_dir:
        from examples.benchmark_plot import run_all_plots

        labels = [t[:20] + "..." if len(t) > 20 else t for t in PROMPTS]
        run_all_plots(all_metrics, labels, [len(p) for p in PROMPTS], args.output_dir)
        print(f"[bench] Charts saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
