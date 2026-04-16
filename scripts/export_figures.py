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
TASKS_MULTICLASS = ["tonal", "articulation", "legato"]
TASKS_BINARY = ["pizzicato", "harmonic", "spiccato", "legato"]
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


def _is_binary_variant(data: dict) -> bool:
    """Detect whether a result dict uses binary technique flags."""
    for ds in DATASETS:
        if ds in data:
            return "pizzicato" in data[ds] or "spiccato" in data[ds]
    return False


def _get_tasks(data: dict) -> list[str]:
    """Return the task list appropriate for this variant."""
    return TASKS_BINARY if _is_binary_variant(data) else TASKS_MULTICLASS


def _split_results(results: dict) -> tuple[dict, dict]:
    """Split results into (multiclass_results, binary_results)."""
    mc, bn = {}, {}
    for var, data in results.items():
        if _is_binary_variant(data):
            bn[var] = data
        else:
            mc[var] = data
    return mc, bn


def _ensure_accuracy_key(d: dict) -> dict:
    """If a per-class dict has 'event_accuracy' but no 'accuracy', copy it over."""
    if "accuracy" not in d and "event_accuracy" in d:
        d = dict(d)
        d["accuracy"] = d["event_accuracy"]
    return d


def _normalize_binary_for_unified(data: dict) -> dict:
    """Convert binary experiment data to multiclass-compatible structure for unified plotting.

    Mapping:
      tonal       = avg(pizzicato, harmonic) for summary metrics
      articulation = spiccato
      legato       = legato  (negative→sustained, positive→bow_change)

    Per-class:
      tonal/pizzicato  ← pizzicato/positive
      tonal/harmonics  ← harmonic/positive
      tonal/none       ← tonal_none_joint if present (GT no pizz & no harm; correct = both preds 0)
      # tonal/openstring — omitted from figures (see plot_per_class_accuracy skip)
      articulation/none     ← spiccato/negative
      articulation/spiccato ← spiccato/positive
      legato/sustained  ← legato/negative
      legato/bow_change ← legato/positive
    """
    _NA = {"accuracy": np.nan, "support": 0}
    out = {}
    for ds in DATASETS:
        if ds not in data:
            continue
        src = data[ds]
        pizz = _ensure_accuracy_key(src.get("pizzicato", {}))
        harm = _ensure_accuracy_key(src.get("harmonic", {}))
        spic = _ensure_accuracy_key(src.get("spiccato", {}))
        leg = _ensure_accuracy_key(src.get("legato", {}))

        tonal = {}
        for m in SUMMARY_METRICS:
            vals = [v for v in [pizz.get(m), harm.get(m)] if v is not None]
            tonal[m] = sum(vals) / len(vals) if vals else 0
        joint_none = src.get("tonal_none_joint") or {}
        none_pc = (
            {"accuracy": joint_none["accuracy"], "support": joint_none["support"]}
            if joint_none.get("support", 0) > 0
            else dict(_NA)
        )
        tonal["per_class"] = {
            "none": none_pc,
            "pizzicato": _ensure_accuracy_key(
                pizz.get("per_class", {}).get("positive", {"accuracy": 0, "support": 0})),
            "harmonics": _ensure_accuracy_key(
                harm.get("per_class", {}).get("positive", {"accuracy": 0, "support": 0})),
            # "openstring": dict(_NA),  # omitted from figures
        }

        artic = {m: spic.get(m, 0) for m in SUMMARY_METRICS}
        artic["per_class"] = {
            "none": _ensure_accuracy_key(
                spic.get("per_class", {}).get("negative", {"accuracy": 0, "support": 0})),
            "spiccato": _ensure_accuracy_key(
                spic.get("per_class", {}).get("positive", {"accuracy": 0, "support": 0})),
        }

        legato = {m: leg.get(m, 0) for m in SUMMARY_METRICS}
        legato["per_class"] = {
            "sustained": _ensure_accuracy_key(
                leg.get("per_class", {}).get("negative", {"accuracy": 0, "support": 0})),
            "bow_change": _ensure_accuracy_key(
                leg.get("per_class", {}).get("positive", {"accuracy": 0, "support": 0})),
        }

        out[ds] = {
            "tonal": tonal,
            "articulation": artic,
            "legato": legato,
            "gate_distribution": src.get("gate_distribution", {}),
            "total_active_frames": src.get("total_active_frames", 0),
        }

    if "checkpoint" in data:
        out["checkpoint"] = data["checkpoint"]
    return out


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


def plot_summary_metrics(results: dict, out_dir: Path, tasks: list[str] = None,
                         suffix: str = ""):
    """Grouped bar chart: summary metrics per task, one figure per dataset."""
    if not results:
        return
    tasks = tasks or _get_tasks(next(iter(results.values())))
    variants = list(results.keys())
    n_variants = len(variants)

    for ds in DATASETS:
        if ds not in next(iter(results.values())):
            continue

        fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.5), sharey=True)
        if len(tasks) == 1:
            axes = [axes]

        for ax, task in zip(axes, tasks):
            metrics_to_plot = ["accuracy", "macro_f1", "weighted_f1"]
            n_metrics = len(metrics_to_plot)
            x = np.arange(n_metrics)
            total_width = 0.75
            bar_w = total_width / n_variants

            for i, var in enumerate(variants):
                task_data = results[var].get(ds, {}).get(task, {})
                vals = [task_data.get(m, task_data.get("event_accuracy", 0))
                        if m == "accuracy" else task_data.get(m, 0)
                        for m in metrics_to_plot]
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
        tag = " (Binary Flags)" if suffix else ""
        fig.suptitle(f"Summary Metrics — {DATASET_DISPLAY[ds]}{tag}", fontsize=13, fontweight="bold", y=1.08)
        fig.tight_layout()
        fname = f"summary_metrics_{ds}{suffix}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)


def plot_per_class_accuracy(results: dict, out_dir: Path, tasks: list[str] = None,
                            suffix: str = ""):
    """Per-class accuracy comparison across variants, one figure per dataset."""
    if not results:
        return
    tasks = tasks or _get_tasks(next(iter(results.values())))
    variants = list(results.keys())
    n_variants = len(variants)

    for ds in DATASETS:
        if ds not in next(iter(results.values())):
            continue

        all_classes = []
        task_labels = []
        for task in tasks:
            per_class = results[variants[0]].get(ds, {}).get(task, {}).get("per_class", {})
            for cls in per_class:
                # Omit tonal openstring from per-class accuracy figures
                if task == "tonal" and cls == "openstring":
                    continue
                all_classes.append((task, cls))
                task_labels.append(f"{task}\n{cls}")

        if not all_classes:
            continue

        n_classes = len(all_classes)
        fig, ax = plt.subplots(figsize=(max(6, n_classes * 1.2), 5))
        x = np.arange(n_classes)
        total_width = 0.75
        bar_w = total_width / n_variants

        for i, var in enumerate(variants):
            vals = []
            for task, cls in all_classes:
                _pc = results[var].get(ds, {}).get(task, {}).get("per_class", {}).get(cls, {})
                acc = _pc.get("accuracy", _pc.get("event_accuracy", 0))
                if acc is None or (isinstance(acc, float) and np.isnan(acc)):
                    acc = np.nan
                vals.append(acc)
            offset = (i - (n_variants - 1) / 2) * bar_w
            safe_vals = [0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in vals]
            bars = ax.bar(x + offset, safe_vals, bar_w, label=var, color=PALETTE[i % len(PALETTE)],
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v is not None and not (isinstance(v, float) and np.isnan(v)) and v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Accuracy")
        tag = " (Binary Flags)" if suffix else ""
        ax.set_title(f"Per-Class Accuracy — {DATASET_DISPLAY[ds]}{tag}", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fname = f"per_class_accuracy_{ds}{suffix}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)


def plot_gate_distribution(results: dict, out_dir: Path, tasks: list[str] = None,
                           suffix: str = ""):
    """Stacked bar chart of gate distributions (global) per task, one figure per (dataset, variant)."""
    if not results:
        return
    tasks = tasks or _get_tasks(next(iter(results.values())))
    variants = list(results.keys())

    for ds in DATASETS:
        if ds not in next(iter(results.values())):
            continue

        for var in variants:
            gate_data = results[var].get(ds, {}).get("gate_distribution")
            if not gate_data:
                continue

            tasks_with_gates = [t for t in tasks if t in gate_data and "global" in gate_data[t]]
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
            tag = " [bin]" if suffix else ""
            ax.set_title(f"Gate Distribution (Global) — {DATASET_DISPLAY[ds]} / {var}{tag}",
                         fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
            fig.tight_layout()
            fname = f"gate_distribution_{ds}_{var}{suffix}.png"
            fig.savefig(out_dir / fname, bbox_inches="tight")
            plt.close(fig)


def plot_gate_by_class(results: dict, out_dir: Path, tasks: list[str] = None,
                       suffix: str = ""):
    """Per-class gate distribution heatmap, one figure per (dataset, variant, task)."""
    if not results:
        return
    tasks = tasks or _get_tasks(next(iter(results.values())))
    variants = list(results.keys())

    for ds in DATASETS:
        if ds not in next(iter(results.values())):
            continue

        for var in variants:
            gate_data = results[var].get(ds, {}).get("gate_distribution")
            if not gate_data:
                continue

            for task in tasks:
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
                tag = " [bin]" if suffix else ""
                ax.set_title(f"Gate by Class — {DATASET_DISPLAY[ds]} / {var} / {task.capitalize()}{tag}",
                             fontsize=10, fontweight="bold")
                fig.tight_layout()
                fname = f"gate_by_class_{ds}_{var}_{task}{suffix}.png"
                fig.savefig(out_dir / fname, bbox_inches="tight")
                plt.close(fig)


def plot_macro_f1_radar(results: dict, out_dir: Path, tasks: list[str] = None,
                        suffix: str = ""):
    """Radar chart comparing macro F1 across tasks for each variant, one per dataset."""
    if not results:
        return
    tasks = tasks or _get_tasks(next(iter(results.values())))
    variants = list(results.keys())

    for ds in DATASETS:
        if ds not in next(iter(results.values())):
            continue

        labels = [t.capitalize() for t in tasks]
        n = len(tasks)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        for i, var in enumerate(variants):
            vals = [results[var].get(ds, {}).get(t, {}).get("macro_f1", 0) for t in tasks]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", linewidth=1.8, label=var, color=PALETTE[i % len(PALETTE)])
            ax.fill(angles, vals, alpha=0.1, color=PALETTE[i % len(PALETTE)])

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0, 1)
        tag = " (Binary Flags)" if suffix else ""
        ax.set_title(f"Macro F1 — {DATASET_DISPLAY[ds]}{tag}", fontsize=13, fontweight="bold", y=1.08)
        ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), framealpha=0.9)
        fig.tight_layout()
        fname = f"macro_f1_radar_{ds}{suffix}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export evaluation results to figures.")
    parser.add_argument("--experiment", type=str, default="ablation_expert_composition_v2",
                        help="Experiment folder name under results/ (e.g. ablation_expert_composition_v2)")
    parser.add_argument("--variants", type=str, default=["baseline", "no_moe", "binary_baseline", "detach_cues", "hierarchical", "crnn_moe", "binary_baseline_event_level", "freeze_trans", "mert_baseline"],
                        help='JSON array of variant names to include, e.g. \'["baseline", "binary_baseline"]\'')
    args = parser.parse_args()

    setup_style()

    exp_dir = RESULTS_DIR / args.experiment
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    out_dir = exp_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    results = load_experiment(exp_dir)
    print(f"Loaded {len(results)} variants: {list(results.keys())}")

    if args.variants is not None:
        selected = args.variants if isinstance(args.variants, list) else json.loads(args.variants)
        missing = [v for v in selected if v not in results]
        if missing:
            print(f"Warning: variants not found and skipped: {missing}")
        results = {k: v for k, v in results.items() if k in selected}
        print(f"Filtered to {len(results)} variants: {list(results.keys())}")
        if not results:
            print("No matching variants — nothing to plot.")
            return

    mc_results, bin_results = _split_results(results)

    # Build unified results: multi-class as-is + binary normalized to multiclass structure
    unified = dict(mc_results)
    for name, data in bin_results.items():
        unified[name] = _normalize_binary_for_unified(data)

    if unified:
        parts = []
        if mc_results:
            parts.append(f"{len(mc_results)} multi-class")
        if bin_results:
            parts.append(f"{len(bin_results)} binary (mapped)")
        print(f"\nUnified comparison: {' + '.join(parts)}")

        print("  Summary metrics...")
        plot_summary_metrics(unified, out_dir, tasks=TASKS_MULTICLASS)

        print("  Per-class accuracy...")
        plot_per_class_accuracy(unified, out_dir, tasks=TASKS_MULTICLASS)

        print("  Macro F1 radar...")
        plot_macro_f1_radar(unified, out_dir, tasks=TASKS_MULTICLASS)

    # Gate plots: multi-class variants (tasks = tonal/articulation/legato)
    if mc_results:
        print(f"\nGate plots for {len(mc_results)} multi-class variants...")
        plot_gate_distribution(mc_results, out_dir, tasks=TASKS_MULTICLASS)
        plot_gate_by_class(mc_results, out_dir, tasks=TASKS_MULTICLASS)

    # Gate plots: binary variants (tasks = pizzicato/harmonic/spiccato/legato)
    if bin_results:
        print(f"\nGate plots for {len(bin_results)} binary variants...")
        plot_gate_distribution(bin_results, out_dir, tasks=TASKS_BINARY, suffix="_binary")
        plot_gate_by_class(bin_results, out_dir, tasks=TASKS_BINARY, suffix="_binary")

    generated = list(out_dir.glob("*.png"))
    print(f"\nDone — {len(generated)} figures saved to {out_dir}/")
    for f in sorted(generated):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
