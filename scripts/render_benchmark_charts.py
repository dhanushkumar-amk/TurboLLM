"""Render light-theme PNG charts, one image per metric plus a one-page overview.

Usage: python scripts/render_benchmark_charts.py benchmarks/results/latest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/charts"))
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        parser.error(f"matplotlib and numpy are required: {exc}")

    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    runs = report.get("runs", [])
    if not runs:
        parser.error("report has no benchmark runs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    engines = list(dict.fromkeys(row["engine"] for row in runs))
    palette = ["#087f6b", "#829317", "#8055a5", "#c46a32"]
    colors = {engine: palette[i % len(palette)] for i, engine in enumerate(engines)}
    plt.rcParams.update({"figure.facecolor": "#f7faf8", "axes.facecolor": "#ffffff",
                         "axes.edgecolor": "#d7e1dc", "axes.labelcolor": "#34463e",
                         "xtick.color": "#63736c", "ytick.color": "#63736c",
                         "text.color": "#1e3028", "grid.color": "#dfe7e3",
                         "font.family": "DejaVu Sans"})

    def save(fig, name: str) -> None:
        fig.savefig(args.out_dir / name, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def unavailable(ax, text: str) -> None:
        ax.text(.5, .58, "AWAITING MEASUREMENT", ha="center", va="center",
                transform=ax.transAxes, color="#52665d", fontsize=12, fontweight="bold")
        ax.text(.5, .43, text, ha="center", va="center", wrap=True,
                transform=ax.transAxes, color="#77857f", fontsize=9, linespacing=1.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def make_line_metric(key: str, title: str, unit: str, filename: str, percentiles: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        found = False
        for engine in engines:
          workload_names=sorted({r.get("workload","default") for r in runs if r["engine"]==engine})
          for workload_name in workload_names:
            selected = sorted((r for r in runs if r["engine"] == engine and r.get("workload","default")==workload_name), key=lambda r: r["concurrency"])
            label=f"{engine} · {workload_name}" if len(workload_names)>1 else engine
            if percentiles:
                xs, p50, p95, p99 = [], [], [], []
                medians_only = True
                for row in selected:
                    values = row.get("summary", {}).get(key) or {}
                    if values.get("p50") is not None:
                        xs.append(row["concurrency"]); p50.append(values["p50"])
                        p95.append(values.get("p95")); p99.append(values.get("p99"))
                        if values.get("p95") is not None and values.get("p99") is not None:
                            medians_only = False
                if xs:
                    found = True
                    label_suffix = " (median only)" if medians_only else ""
                    ax.plot(xs, p50, "o-", color=colors[engine], linewidth=2.2, label=f"{label} p50{label_suffix}")
                    if not medians_only:
                        ax.plot(xs, p95, "--", color=colors[engine], linewidth=1.6, label=f"{label} p95")
                        ax.plot(xs, p99, ":", color=colors[engine], linewidth=1.8, label=f"{label} p99")
            else:
                xs, ys = [], []
                for row in selected:
                    value = row.get("summary", {}).get(key)
                    if value is not None:
                        xs.append(row["concurrency"]); ys.append(value)
                if xs:
                    found = True
                    ax.plot(xs, ys, "o-", color=colors[engine], linewidth=2.2, markersize=6, label=label)
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=14)
        historical = any(r.get("workload_id") for r in runs)
        ax.set_xlabel("Historical workload (short → long)" if historical else "Concurrent requests")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=.8)
        ax.spines[["top", "right"]].set_visible(False)
        if found:
            ticks = sorted({r["concurrency"] for r in runs})
            if historical:
                labels = []
                for tick in ticks:
                    workload_id = next((r.get("workload_id", "") for r in runs if r["concurrency"] == tick), "")
                    labels.append("Short prompt / 8 output" if workload_id.startswith("short") else "Long prompt / 96 output")
                ax.set_xticks(ticks, labels, rotation=12, ha="right")
            else:
                ax.set_xticks(ticks)
            ax.legend(frameon=False, ncol=min(3, max(1, len(ax.get_legend_handles_labels()[0]))))
        else:
            unavailable(ax, "measurement unavailable in source report")
        save(fig, filename)

    make_line_metric("ttft_ms", "Time to first token (TTFT)", "Milliseconds", "ttft.png", True)
    make_line_metric("tpot_ms", "TPOT estimate (ITL proxy)", "Milliseconds", "itl_tpot.png", True)
    make_line_metric("e2e_ms", "End-to-end request latency", "Milliseconds", "e2e_latency.png", True)
    make_line_metric("output_tokens_per_sec", "System output throughput", "Output tokens / second", "output_tokens_per_second.png")
    make_line_metric("requests_per_sec", "Request throughput", "Requests / second", "requests_per_second.png")
    make_line_metric("goodput_requests_per_sec", "SLO goodput", "Requests / second meeting SLO", "goodput.png")

    efficiency = report.get("efficiency", {})
    efficiency_metrics = [
        ("gpu_utilization_percent", "GPU utilization", "%"),
        ("memory_bandwidth_utilization_percent", "Memory bandwidth utilization (MBU)", "% of peak bandwidth"),
        ("prefill_mfu_percent", "Prefill MFU", "% model FLOPs utilization"),
        ("kv_cache_usage_percent", "KV cache usage", "% of cache capacity"),
        ("prefix_cache_hit_rate_percent", "Prefix cache hit rate", "%"),
        ("cost_per_million_tokens", "Cost per million tokens", report.get("workload", {}).get("currency", "currency")),
    ]
    for key, title, unit in efficiency_metrics:
        values = efficiency.get(key)
        has_values = isinstance(values, (int, float)) or (isinstance(values, dict) and any(
            (isinstance(v, (int, float)) and v is not None) or
            (isinstance(v, dict) and any(isinstance(n, (int, float)) and n is not None for n in v.values()))
            for v in values.values()))
        if not has_values:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        guidance = {
            "gpu_utilization_percent": "Capture per-process GPU utilization with NVML or DCGM during each engine run.",
            "memory_bandwidth_utilization_percent": "Requires GPU memory-controller bytes/sec divided by peak bandwidth (Nsight/DCGM).",
            "prefill_mfu_percent": "Requires prefill FLOP estimate and measured prefill duration for this model.",
            "kv_cache_usage_percent": "Requires each server's active KV blocks/tokens and cache capacity counters.",
            "prefix_cache_hit_rate_percent": "Requires prefix-cache hit and eligible-token counters; enable prefix caching.",
            "cost_per_million_tokens": "Pass --gpu-hourly-cost with your actual GPU + host hourly rate to calculate this chart.",
        }
        if values is None:
            unavailable(ax, guidance[key])
        else:
            if not isinstance(values, dict):
                values = {engine: values for engine in engines[:1]}
            names, nums = [], []
            for name, value in values.items():
                if isinstance(value, dict):
                    for level, number in value.items():
                        if number is not None:
                            names.append(f"{name} C={level}"); nums.append(number)
                elif value is not None:
                    names.append(name); nums.append(value)
            if names:
                bars = ax.bar(names, nums, color=[colors.get(name.split(" C=")[0], "#087f6b") for name in names], width=.55)
                ax.bar_label(bars, fmt="%.2f", padding=4, color="#263a31")
                ax.set_ylabel(unit); ax.grid(axis="y", alpha=.8); ax.set_axisbelow(True)
            else:
                unavailable(ax, guidance[key])
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=14)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, f"{key.replace('_percent','')}.png")

    # One-page image overview: each requested metric retains its own plot area.
    overview_items = [
        ("ttft_ms", "TTFT (ms)", True), ("tpot_ms", "TPOT estimate (ms)", True),
        ("e2e_ms", "E2E latency (ms)", True), ("output_tokens_per_sec", "Output tokens / sec", False),
        ("requests_per_sec", "Requests / sec", False), ("goodput_requests_per_sec", "Goodput (req / sec)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.flat
    for ax, (key, title, is_latency) in zip(axes, overview_items):
        found = False
        for engine in engines:
          workload_names=sorted({r.get("workload","default") for r in runs if r["engine"]==engine})
          for workload_name in workload_names:
            selected = sorted((r for r in runs if r["engine"] == engine and r.get("workload","default")==workload_name), key=lambda r: r["concurrency"])
            label=f"{engine} · {workload_name}" if len(workload_names)>1 else engine
            if is_latency:
                valid = [(r["concurrency"], r["summary"].get(key) or {}) for r in selected]
                valid = [(x,v) for x,v in valid if v.get("p50") is not None]
                if valid:
                    found = True; xs=[p[0] for p in valid]; vals=[p[1] for p in valid]
                    ax.plot(xs,[v["p50"] for v in vals],"o-",color=colors[engine],label=f"{label} p50")
                    if all(v.get("p95") is not None and v.get("p99") is not None for v in vals):
                        ax.plot(xs,[v["p95"] for v in vals],"--",color=colors[engine],label=f"{label} p95")
                        ax.plot(xs,[v["p99"] for v in vals],":",color=colors[engine],label=f"{label} p99")
            else:
                valid=[(r["concurrency"],r["summary"].get(key)) for r in selected if r["summary"].get(key) is not None]
                if valid:
                    found=True;ax.plot([p[0] for p in valid],[p[1] for p in valid],"o-",color=colors[engine],label=label)
        ax.set_title(title,loc="left",fontweight="bold");ax.grid(axis="y",alpha=.7);ax.spines[["top","right"]].set_visible(False)
        if any(r.get("workload_id") for r in runs):
            ticks=sorted({r["concurrency"] for r in runs})
            tick_labels=[]
            for tick in ticks:
                wid=next((r.get("workload_id","") for r in runs if r["concurrency"]==tick),"")
                tick_labels.append("Short workload" if wid.startswith("short") else "Long workload")
            ax.set_xticks(ticks,tick_labels,rotation=12,ha="right")
        if found: ax.legend(frameon=False,fontsize=7)
        else: unavailable(ax,"not available in this report")
    fig.suptitle(report.get("title", "Inference benchmark report"), x=.04, ha="left", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0,.01,1,.96),h_pad=2.2,w_pad=1.8)
    save(fig,"benchmark_overview.png")
    print(f"Wrote individual metric PNGs and benchmark_overview.png to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
