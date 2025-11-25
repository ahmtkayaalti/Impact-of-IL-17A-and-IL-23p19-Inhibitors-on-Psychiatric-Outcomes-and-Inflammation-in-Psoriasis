
"""
Reproducible scripts for Figure 3 and Figure 4
Psoriasis biologics – CRP / NLR / HADS analyses

Requirements
------------
Python >= 3.9
Packages: pandas, numpy, matplotlib, seaborn, scipy

Usage
-----
1. Put this script in the same folder as the Excel file:
     "GÜNCEL PS EXCEL BİRLEŞİK.xlsx"
2. Run:
     python figure_scripts.py
3. The script will save:
     - figure3_crp_hads_scatter.png
     - figure4_dunn_heatmaps.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# General plotting style
# ---------------------------------------------------------------------
sns.set(style="whitegrid", context="talk")
np.random.seed(42)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_data(excel_path: str) -> pd.DataFrame:
    """Load the main dataset and compute delta variables."""
    df = pd.read_excel(excel_path)

    # Strip accidental spaces in column names
    df.columns = df.columns.str.strip()

    # Change scores (6th Month - Baseline)
    df["dCRP"] = df["CRP (6th Month)"] - df["CRP (Baseline)"]
    df["dHADS_D"] = df["HADS-D (6th Month)"] - df["HADS-D (Baseline)"]
    df["dHADS_A"] = df["HADS-A (6th Month)"] - df["HADS-A (Baseline)"]
    df["dPASI"] = df["PASI (6th Month)"] - df["PASI (Baseline)"]

    # PASI90 response (can be adapted if a PASI90 column already exists)
    df["PASI90"] = df["PASI (6th Month)"] <= 0.1 * df["PASI (Baseline)"]

    return df


def plot_figure3_scatter(df: pd.DataFrame, out_path: str = "figure3_crp_hads_scatter.png"):
    """Replicates Figure 3: ΔCRP vs ΔHADS-D and ΔHADS-A by PASI90 status."""
    # Prepare colors and labels
    palette = {True: "tab:blue", False: "tab:red"}
    label_map = {True: "PASI90", False: "Non-PASI90"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=False)

    # (a) ΔCRP vs ΔHADS-D
    ax = axes[0]
    for pasi90_value in [False, True]:
        subset = df[df["PASI90"] == pasi90_value]
        sns.regplot(
            data=subset,
            x="dCRP",
            y="dHADS_D",
            ax=ax,
            scatter=True,
            scatter_kws={"alpha": 0.7, "s": 30},
            line_kws={"linewidth": 2},
            color=palette[pasi90_value],
        )
    ax.set_title("(a) Correlation between ΔCRP and ΔHADS-D by PASI90 status")
    ax.set_xlabel("CRP reduction (ΔCRP)")
    ax.set_ylabel("HADS-D reduction (ΔHADS-D)")

    handles = [
        plt.Line2D([0], [0], color=palette[False], marker="o", linestyle="", label=label_map[False]),
        plt.Line2D([0], [0], color=palette[True], marker="o", linestyle="", label=label_map[True]),
    ]
    ax.legend(handles=handles, title="PASI90 response")

    # (b) ΔCRP vs ΔHADS-A
    ax = axes[1]
    for pasi90_value in [False, True]:
        subset = df[df["PASI90"] == pasi90_value]
        sns.regplot(
            data=subset,
            x="dCRP",
            y="dHADS_A",
            ax=ax,
            scatter=True,
            scatter_kws={"alpha": 0.7, "s": 30},
            line_kws={"linewidth": 2},
            color=palette[pasi90_value],
        )
    ax.set_title("(b) Correlation between ΔCRP and ΔHADS-A by PASI90 status")
    ax.set_xlabel("CRP reduction (ΔCRP)")
    ax.set_ylabel("HADS-A reduction (ΔHADS-A)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_pvalue_stars(p: float) -> str:
    """Return p-value as string with significance stars.""
    if p < 0.001:
        return f"{p:.3f}***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


def plot_dunn_heatmap(p_mat: np.ndarray, labels, title: str, ax):
    """Plot a symmetric heatmap of pairwise Dunn p-values.

    p_mat must be a square matrix with np.nan on the diagonal.
    """
    # Mask diagonal
    mask = np.eye(len(labels), dtype=bool)

    sns.heatmap(
        p_mat,
        mask=mask,
        annot=[[add_pvalue_stars(p) if not np.isnan(p) else "" for p in row] for row in p_mat],
        fmt="s",
        cmap="coolwarm_r",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Dunn pairwise p-value"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, ha="right")
    ax.tick_params(axis="y", rotation=0)


def plot_figure4_heatmaps(out_path: str = "figure4_dunn_heatmaps.png"):
    """Replicates Figure 4: pairwise Dunn–Bonferroni p-value heatmaps.

    IMPORTANT: The p-value matrices below must be filled with the
    study-specific Dunn–Bonferroni results. The shapes and labels
    correspond to the four biologic agents used in the study.
    """
    labels = ["Secukinumab", "Ixekizumab", "Risankizumab", "Guselkumab"]

    # ------------------------------------------------------------------
    # Replace the example matrices below with the actual Dunn p-values
    # from the analyses (upper or lower triangle; diagonal as np.nan).
    # ------------------------------------------------------------------

    # Example structure for DLQI reduction
    dlqi_p = np.array([
        [np.nan, 0.246, 0.240, 0.004],
        [0.246, np.nan, 0.932, 0.038],
        [0.240, 0.932, np.nan, 0.059],
        [0.004, 0.038, 0.059, np.nan],
    ])

    # Example structure for CRP reduction
    crp_p = np.array([
        [np.nan, 0.830, 0.210, 0.066],
        [0.830, np.nan, 0.210, 0.073],
        [0.210, 0.210, np.nan, 0.600],
        [0.066, 0.073, 0.600, np.nan],
    ])

    # Example structure for NLR reduction
    nlr_p = np.array([
        [np.nan, 0.830, 0.390, 0.400],
        [0.830, np.nan, 0.490, 0.540],
        [0.390, 0.490, np.nan, 0.940],
        [0.400, 0.540, 0.940, np.nan],
    ])

    fig, axes = plt.subplots(3, 1, figsize=(6, 16))

    plot_dunn_heatmap(dlqi_p, labels, "(a) DLQI reduction", axes[0])
    plot_dunn_heatmap(crp_p, labels, "(b) CRP reduction", axes[1])
    plot_dunn_heatmap(nlr_p, labels, "(c) NLR reduction", axes[2])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Path to the main Excel file
    excel_path = "GÜNCEL PS EXCEL BİRLEŞİK.xlsx"

    # Load and prepare data
    df = load_data(excel_path)

    # Figure 3: scatter plots with regression lines
    plot_figure3_scatter(df)

    # Figure 4: Dunn–Bonferroni heatmaps
    plot_figure4_heatmaps()

    print("Figures saved as 'figure3_crp_hads_scatter.png' and 'figure4_dunn_heatmaps.png'.")
