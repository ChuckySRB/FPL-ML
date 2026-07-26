"""Create compact EDA figures for the corrected academic dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "tier2_fixed_2020-21_to_2024_25"
FIGURE_DIR = ROOT / "outputs" / "figures"


def main() -> None:
    """Generate target, drift, and correlation figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train_full.csv", low_memory=False)
    test = pd.read_csv(DATA_DIR / "test_full.csv", low_memory=False)
    data = pd.concat([train, test], ignore_index=True)
    data = data[data["position_label"].isin(["GK", "DEF", "MID", "FWD"])]

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    clipped = data["total_points"].clip(-2, 15)
    sns.histplot(clipped, bins=35, color="#2563eb", ax=ax)
    ax.axvline(
        data["total_points"].mean(),
        color="#dc2626",
        linestyle="--",
        label=f"Mean = {data['total_points'].mean():.2f}",
    )
    ax.set(
        title="FPL points are zero-heavy with a long positive tail",
        xlabel="Points per player-fixture (clipped at 15)",
        ylabel="Rows",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_target_distribution_fixed.png", dpi=180)
    plt.close(fig)

    season_stats = (
        data.groupby("season", as_index=False)["total_points"]
        .agg(mean="mean", std="std", median="median")
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(
        season_stats["season"],
        season_stats["mean"],
        yerr=season_stats["std"],
        fmt="o-",
        capsize=5,
        color="#059669",
        linewidth=2,
    )
    ax.plot(
        season_stats["season"],
        season_stats["median"],
        "s--",
        color="#7c3aed",
        label="Median",
    )
    ax.set(
        title="Target statistics remain broadly stable across seasons",
        xlabel="Season",
        ylabel="FPL points",
    )
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_season_drift_fixed.png", dpi=180)
    plt.close(fig)

    feature_cols = [
        "total_points",
        "form_last_3",
        "form_last_5",
        "minutes_last_3",
        "ict_index_last_3",
        "opponent_difficulty",
        "price",
        "selected_pct",
        "team_strength",
        "opponent_strength",
    ]
    corr = data[feature_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title("Correlation of core predictors and target")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_correlation_fixed.png", dpi=180)
    plt.close(fig)

    season_stats.to_csv(
        ROOT / "outputs" / "results" / "eda_season_summary_fixed.csv",
        index=False,
    )
    print("Saved EDA figures to outputs/figures")


if __name__ == "__main__":
    main()
