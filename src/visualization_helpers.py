from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def _save_and_show(save_path=None, dpi: int = 300):
    """
    Saves the current figure to disk if a path is given, then displays it.

    Args:
        save_path: File path to save the figure. If None, only displays.
        dpi: Resolution for the saved image
    Returns:
        None
    """
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    plt.tight_layout()
    plt.show()


def visualize_multiple_comparisons(results_dict: Dict[str, pd.DataFrame], save_path=None):
    """
    Plots a grouped bar chart comparing performance and error patterns across experiments.

    Args:
        results_dict: Mapping of experiment labels to their result DataFrames
        save_path: File path to save the figure. If None, only displays.
    Returns:
        None
    """
    def get_metrics(df):
        acc = df["is_correct"].mean() * 100
        rep_in = df["repeat_input"].mean() * 100
        rep_rel = df["repeat_relation"].mean() * 100
        return [acc, rep_in, rep_rel, acc + rep_in + rep_rel]

    categories = ["Accuracy", "Repeat Input", "Repeat Relation", "Total (Tracked)"]
    labels = list(results_dict.keys())
    all_values = [get_metrics(df) for df in results_dict.values()]

    x = np.arange(len(categories))
    n_groups = len(labels)
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(16, 9))

    for i, (label, values) in enumerate(zip(labels, all_values)):
        offset = (i - (n_groups - 1) / 2) * width
        rects = ax.bar(x + offset, values, width, label=label, alpha=0.8)

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Multi-Condition Performance & Error Pattern Comparison", fontsize=16, pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    _save_and_show(save_path=save_path)


def plot_patching_heatmap(df: pd.DataFrame, title: str = "Activation Patching Scores", top_n: int = 10, save_path=None):
    """
    Plots a layer-by-head heatmap of activation patching scores with top candidates highlighted.

    Args:
        df: DataFrame with 'layer', 'head', and 'mean_patch_score' columns
        title: Title for the plot
        top_n: Number of top-scoring heads to highlight with red borders
        save_path: File path to save the figure. If None, only displays.
    Returns:
        None
    """
    plt.figure(figsize=(16, 10))
    heatmap_data = df.pivot(index="layer", columns="head", values="mean_patch_score")
    ax = sns.heatmap(heatmap_data, cmap="RdBu", center=0, robust=True)

    if top_n > 0:
        top_candidates = df.nlargest(top_n, "mean_patch_score")
        for _, row in top_candidates.iterrows():
            l = int(row["layer"])
            h = int(row["head"])

            if l in heatmap_data.index and h in heatmap_data.columns:
                y_idx = heatmap_data.index.get_loc(l)
                x_idx = heatmap_data.columns.get_loc(h)
                rect = patches.Rectangle((x_idx, y_idx), 1, 1, fill=False, edgecolor="red", lw=3)
                ax.add_patch(rect)

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Head Index", fontsize=14)
    plt.ylabel("Layer Index", fontsize=14)
    ax.invert_yaxis()

    _save_and_show(save_path=save_path)


def plot_intervention_results(summary_df: pd.DataFrame, model_name: str = "Model", metric: str = "patching_score"):
    """
    Plots a line chart of restoration scores by intervention layer.

    Args:
        summary_df: DataFrame indexed by intervention_layer with score columns
        model_name: Model name to display in the title
        metric: Column name for the y-axis metric
    Returns:
        None
    """
    df_plot = summary_df.reset_index().sort_values("intervention_layer")

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_plot, x="intervention_layer", y=metric, marker="o", color="purple")
    plt.title(f"Restoration Score by Layer ({model_name})")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Patching Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_intervention_results_v2(summary_df: pd.DataFrame, model_name: str = "Model"):
    """
    Convenience wrapper for plot_intervention_results with patching_score metric.

    Args:
        summary_df: DataFrame indexed by intervention_layer with score columns
        model_name: Model name to display in the title
    Returns:
        None
    """
    plot_intervention_results(summary_df, model_name=model_name, metric="patching_score")


def plot_combined_intervention_results(
    relation_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    save_path=None,
    model_name: str = "Model",
):
    """
    Plots relation vector vs. random control restoration scores on the same chart.

    Args:
        relation_summary: Summary DataFrame for the relation vector condition
        random_summary: Summary DataFrame for the random control condition
        save_path: File path to save the figure. If None, only displays.
        model_name: Model name to display in the title
    Returns:
        None
    """
    df_relation = relation_summary.reset_index().sort_values("intervention_layer")
    df_random = random_summary.reset_index().sort_values("intervention_layer")

    df_relation["Type"] = "Relation Vector"
    df_random["Type"] = "Random Control"

    combined_df = pd.concat([df_relation, df_random])

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=combined_df,
        x="intervention_layer",
        y="patching_score",
        hue="Type",
        marker="o",
        palette={"Relation Vector": "purple", "Random Control": "gray"},
    )

    plt.title(f"Restoration Score Comparison: Relation vs. Random ({model_name})")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Patching Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    _save_and_show(save_path=save_path)


def plot_intervention_accuracy_comparison(
    baseline_accuracy: float,
    intervention_accuracy: float,
    layer_label,
    alpha_value: float,
    save_path=None,
    model_name: Optional[str] = None,
):
    """
    Plots a bar chart comparing baseline vs. intervention accuracy.

    Args:
        baseline_accuracy: Accuracy (%) without intervention
        intervention_accuracy: Accuracy (%) with intervention applied
        layer_label: Layer identifier to display in the title
        alpha_value: Scaling factor used for the intervention
        save_path: File path to save the figure. If None, only displays.
        model_name: Model name to display in the title
    Returns:
        None
    """
    comparison_data = {
        "Condition": ["Baseline (None Relation)", "Intervention (None + Vector)"],
        "Accuracy (%)": [baseline_accuracy, intervention_accuracy],
    }
    df_plot = pd.DataFrame(comparison_data)

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x="Condition", y="Accuracy (%)", data=df_plot, palette=["gray", "purple"])

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

    title = f"Accuracy Comparison: Before vs After Intervention\n(Layer: {layer_label}, Alpha: {alpha_value})"
    if model_name:
        title = f"{title}\n{model_name}"

    plt.title(title, fontsize=14)
    plt.ylim(0, max(intervention_accuracy, baseline_accuracy) + 10)
    plt.ylabel("Top-1 Accuracy (%)")

    _save_and_show(save_path=save_path)

    
def plot_layer_intervention_accuracy(
    layer_metrics: pd.DataFrame,
    model_name: str = "Model",
    save_path=None,
):
    """
    Plots per-layer intervention accuracy for input/output prediction,
    with baseline (NaN-indexed rows) shown as dashed horizontal lines.

    Args:
        layer_metrics: DataFrame indexed by intervention_layer with prediction columns
        model_name: Model name to display in the title
        save_path: File path to save the figure. If None, only displays.
    Returns:
        None
    """
    sns.set_theme(style="whitegrid")

    baseline_mask = layer_metrics.index.isna()
    baseline_metrics = layer_metrics[baseline_mask]
    numeric_layer_metrics = layer_metrics[~baseline_mask]

    plt.figure(figsize=(12, 7))

    colors = {"input_prediction": "orange", "output_prediction": "purple"}
    labels = {"input_prediction": "Input Prediction", "output_prediction": "Output Prediction"}

    x_axis = numeric_layer_metrics.index.astype(float).astype(int)

    for metric in ["input_prediction", "output_prediction"]:
        y_axis = numeric_layer_metrics[metric].values
        sns.lineplot(x=x_axis, y=y_axis, marker="o", label=f"{labels[metric]} (Intervention)", color=colors[metric])

        if not baseline_metrics.empty:
            baseline_val = baseline_metrics[metric].values[0]
            if not np.isnan(baseline_val):
                plt.axhline(
                    y=baseline_val,
                    color=colors[metric],
                    linestyle="--",
                    alpha=0.5,
                    label=f"{labels[metric]} Baseline: {baseline_val:.1f}%",
                )

    plt.title(f"Intervention Accuracy Comparison: {model_name}", fontsize=16)
    plt.xlabel("Intervention Layer Index", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    _save_and_show(save_path=save_path)


def visualize_intervention_scores(scores: torch.Tensor, top_n: int = 10, title: str = "Head Intervention Scores", save_path: Optional[Path] = None):
    """
    Plots a layer-by-head heatmap of intervention scores and highlights top heads.

    Args:
        scores: Intervention scores tensor of shape (num_layers, num_heads)
        top_n: Number of top-scoring heads to highlight with red borders
        title: Title for the plot
        save_path: File path to save the figure. If None, only displays.
    Returns:
        None
    """
    num_layers, num_heads = scores.shape
    
    # Prepare data for plotting
    data = []
    for l in range(num_layers):
        for h in range(num_heads):
            data.append({"layer": l, "head": h, "score": scores[l, h].item()})
    df_scores = pd.DataFrame(data)

    plt.figure(figsize=(16, 10))
    # Pivot the data for the heatmap
    heatmap_data = df_scores.pivot(index="layer", columns="head", values="score")
    
    # Plot heatmap
    ax = sns.heatmap(heatmap_data, cmap="RdBu", center=0, robust=True, cbar_kws={'label': 'Logit Difference'})

    if top_n > 0:
        # Identify top N heads based on score
        top_candidates = df_scores.nlargest(top_n, "score")
        for _, row in top_candidates.iterrows():
            l = int(row["layer"])
            h = int(row["head"])

            # Map layer/head to heatmap indices
            y_idx = heatmap_data.index.get_loc(l)
            x_idx = heatmap_data.columns.get_loc(h)
            
            # Add red rectangle around the top heads
            rect = patches.Rectangle((x_idx, y_idx), 1, 1, fill=False, edgecolor="red", lw=3)
            ax.add_patch(rect)

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Head Index", fontsize=14)
    plt.ylabel("Layer Index", fontsize=14)
    ax.invert_yaxis()
    _save_and_show(save_path=save_path)


def plot_accuracy_barplot(
    dataframes: dict[str, pd.DataFrame],
    save_path=None,
):
    metrics = ["input_prediction", "relation_prediction", "output_prediction"]
    group_labels = ["Input Repeat", "Relation Repeat", "Correct Output"]
    condition_labels = [name.capitalize() for name in dataframes]
    palette = ["#5B9BD5", "#ED7D31", "#70AD47", "#FFC000", "#A855F7", "#EF4444"]
    colors = palette[:len(dataframes)]

    accuracies = np.array([
        [df[m].mean() * 100 for m in metrics]
        for df in dataframes.values()
    ])

    x = np.arange(len(group_labels))
    n = len(condition_labels)
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, color) in enumerate(zip(condition_labels, colors)):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, accuracies[i], width, label=label, color=color, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Prediction Accuracy by Relation Condition", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylim(0, max(accuracies.max() + 15, 50))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine()

    _save_and_show(save_path=save_path)