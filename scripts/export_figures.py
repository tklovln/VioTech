#!/usr/bin/env python3
"""Export evaluation results from a given experiment folder to figures."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DATASETS = ["viotech", "rwc"]
DATASET_DISPLAY = {"viotech": "VioTech", "rwc": "RWC"}
TASKS = ["tonal", "articulation", "legato"]
SUMMARY_METRICS = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
}

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


def setup_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def load_experiment(exp_dir: Path) -> dict[str, dict]:
    """Load all JSON result files from an experiment directory, keyed by variant name."""
    results = {}
    for f in sorted(exp_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        variant = f.stem
        with open(f) as fh:
            results[variant] = json.load(fh)
    return results


def plot_summary_metrics(results: dict, out_dir: Path):
    """Grouped bar chart: summary metrics per task, one figure per dataset."""
    variants = list(results.keys())
    n_variants = len(variants)

    for ds in DATASETS:
        if ds not in next(iter(results.values())):
            continue

        fig, axes = plt.subplots(1, len(TASKS), figsize=(5 * len(TASKS), 4.5), sharey=True)
        if len(TASKS) == 1:
            axes = [axes]

        for ax, task in zip(axes, TASKS):
            metrics_to_plot = ["accuracy", "macro_f1", "weighted_f1"]
            n_metrics = len(metrics_to_plot)
            x = np.arange(n_metrics)
            total_width = 0.75
            bar_w = total_width / n_variants

            for i, var in enumerate(variants):
                vals = [results[var][ds][task].get(m, 0) for m in metrics_to_plot]
                offset = (i - (n_variants - 1) / 2) * bar_w
                bars = ax.bar(x + offset, vals, bar_w, label=var, color=PALETTE[i % len(PALETTE)],
                              edgecolor="white", linewidth=0.5)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

            ax.set_xticks(x)
            ax.set_xticklabels([METRIC_DISPLAY[m] for m in metrics_to_plot])
            ax.set_title(task.capitalize())
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Score" if ax == axes[0] else "")

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(n_variants, 5),
                   bbox_to_anchor=(0.5, 1.02), frameon=False)
        fig.suptitle(f"Summary Metrics — {DATASET_DISPLAY[ds]}", fontsize=13, fontweight="bold", y=1.08)
        fig.tight_layout()
        fig.savefig(out_dir / f"summary_metrics_{ds}.png", bbox_inches="tight")
        plt.close(fig)


def plot_per_class_accuracy(results: dict, out_dir: Path):
    """Per-class accuracy comparison across variants, one figure per dataset."""
    variants = list(results.keys())
    n_variants = len(variants)

    for ds in DATASETS:
        if ds not in next(iter(results.values())):
            continue

        all_classes = []
        task_labels = []
        for task in TASKS:
            sample = results[variants[0]][ds][task]["per_class"]
            for cls in sample:
                all_classes.append((task, cls))
                task_labels.append(f"{task}\n{cls}")

        n_classes = len(all_classes)
        fig, ax = plt.subplots(figsize=(max(6, n_classes * 1.2), 5))
        x = np.arange(n_classes)
        total_width = 0.75
        bar_w = total_width / n_variants

        for i, var in enumerate(variants):
            vals = []
            for task, cls in all_classes:
                acc = results[var][ds][task]["per_class"].get(cls, {}).get("accuracy", 0)
                vals.append(acc)
            offset = (i - (n_variants - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=var, color=PALETTE[i % len(PALETTE)],
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Per-Class Accuracy — {DATASET_DISPLAY[ds]}", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"per_class_accuracy_{ds}.png", bbox_inches="tight")
        plt.close(fig)


def plot_gate_distribution(results: dict, out_dir: Path):
    """Stacked bar chart of gate distributions (global) per task, one figure per (dataset, variant)."""
    variants = list(results.keys())

    for ds in DATASETS:
        if ds not in next(iter(results.values())):
            continue

        for var in variants:
            gate_data = results[var][ds].get("gate_distribution")
            if not gate_data:
                continue

            tasks_with_gates = [t for t in TASKS if t in gate_data and "global" in gate_data[t]]
            if not tasks_with_gates:
                continue

            all_expert_names = set()
            for task in tasks_with_gates:
                all_expert_names.update(gate_data[task]["global"].keys())
            expert_names = sorted(all_expert_names)

            fig, ax = plt.subplots(figsize=(max(4, len(tasks_with_gates) * 2), 4.5))
            x = np.arange(len(tasks_with_gates))
            bottoms = np.zeros(len(tasks_with_gates))

            for j, expert in enumerate(expert_names):
                vals = []
                for task in tasks_with_gates:
                    vals.append(gate_data[task]["global"].get(expert, 0))
                vals = np.array(vals)
                ax.bar(x, vals, 0.5, bottom=bottoms, label=expert,
                       color=PALETTE[j % len(PALETTE)], edgecolor="white", linewidth=0.5)
                for k, (v, b) in enumerate(zip(vals, bottoms)):
                    if v > 3:
                        ax.text(x[k], b + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=7)
                bottoms += vals

            ax.set_xticks(x)
            ax.set_xticklabels([t.capitalize() for t in tasks_with_gates])
            ax.set_ylabel("Gate Weight (%)")
            ax.set_ylim(0, 110)
            ax.set_title(f"Gate Distribution (Global) — {DATASET_DISPLAY[ds]} / {var}",
                         fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
            fig.tight_layout()
            fig.savefig(out_dir / f"gate_distribution_{ds}_{var}.png", bbox_inches="tight")
            plt.close(fig)


def plot_gate_by_class(results: dict, out_dir: Path):
    """Per-class gate distribution heatmap, one figure per (dataset, variant, task)."""
    variants = list(results.keys())

    for ds in DATASETS:
        if ds not in next(iter(results.values())):
            continue

        for var in variants:
            gate_data = results[var][ds].get("gate_distribution")
            if not gate_data:
                continue

            for task in TASKS:
                if task not in gate_data or "by_class" not in gate_data[task]:
                    continue
                by_class = gate_data[task]["by_class"]
                classes = list(by_class.keys())
                if not classes:
                    continue

                expert_names = sorted(by_class[classes[0]].keys())
                matrix = np.array([[by_class[c].get(e, 0) for e in expert_names] for c in classes])

                fig, ax = plt.subplots(figsize=(max(4, len(expert_names) * 1.3), max(2.5, len(classes) * 0.7)))
                im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
                ax.set_xticks(range(len(expert_names)))
                ax.set_xticklabels(expert_names, fontsize=7, rotation=30, ha="right")
                ax.set_yticks(range(len(classes)))
                ax.set_yticklabels(classes, fontsize=8)
                for row in range(len(classes)):
                    for col in range(len(expert_names)):
                        v = matrix[row, col]
                        color = "white" if v > 50 else "black"
                        ax.text(col, row, f"{v:.1f}", ha="center", va="center", fontsize=7, color=color)

                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("Gate Weight (%)", fontsize=8)
                ax.set_title(f"Gate by Class — {DATASET_DISPLAY[ds]} / {var} / {task.capitalize()}",
                             fontsize=10, fontweight="bold")
                fig.tight_layout()
                fig.savefig(out_dir / f"gate_by_class_{ds}_{var}_{task}.png", bbox_inches="tight")
                plt.close(fig)


def plot_macro_f1_radar(results: dict, out_dir: Path):
    """Radar chart comparing macro F1 across tasks for each variant, one per dataset."""
    variants = list(results.keys())

    for ds in DATASETS:
        if ds not in next(iter(results.values())):
            continue

        labels = [t.capitalize() for t in TASKS]
        n = len(TASKS)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        for i, var in enumerate(variants):
            vals = [results[var][ds][t]["macro_f1"] for t in TASKS]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", linewidth=1.8, label=var, color=PALETTE[i % len(PALETTE)])
            ax.fill(angles, vals, alpha=0.1, color=PALETTE[i % len(PALETTE)])

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0, 1)
        ax.set_title(f"Macro F1 — {DATASET_DISPLAY[ds]}", fontsize=13, fontweight="bold", y=1.08)
        ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"macro_f1_radar_{ds}.png", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export evaluation results to figures.")
    parser.add_argument("experiment", type=str,
                        help="Experiment folder name under results/ (e.g. ablation_expert_composition_v2)")
    args = parser.parse_args()

    setup_style()

    exp_dir = RESULTS_DIR / args.experiment
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    out_dir = exp_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    results = load_experiment(exp_dir)
    print(f"Loaded {len(results)} variants: {list(results.keys())}")

    print("Plotting summary metrics...")
    plot_summary_metrics(results, out_dir)

    print("Plotting per-class accuracy...")
    plot_per_class_accuracy(results, out_dir)

    print("Plotting gate distribution (global)...")
    plot_gate_distribution(results, out_dir)

    print("Plotting gate distribution (by class)...")
    plot_gate_by_class(results, out_dir)

    print("Plotting macro F1 radar...")
    plot_macro_f1_radar(results, out_dir)

    generated = list(out_dir.glob("*.png"))
    print(f"\nDone — {len(generated)} figures saved to {out_dir}/")
    for f in sorted(generated):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
