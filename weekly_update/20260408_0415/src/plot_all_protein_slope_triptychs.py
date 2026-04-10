import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PATIENT_FEATURES_PATH = Path(
    "/workspace/weekly_update/20260408_0415/results/slope_based_selection/patient_by_slope_features.csv"
)
DEFAULT_RANKING_PATH = Path(
    "/workspace/weekly_update/20260408_0415/results/slope_based_selection/slope_feature_ranking.csv"
)
DEFAULT_OUTPUT_BASE_DIR = Path(
    "/workspace/weekly_update/20260408_0415/results/slope_based_selection"
)

GROUP_ORDER = ["cvt", "ctrl", "mnt"]
GROUP_PALETTE = {
    "cvt": "#d62728",
    "ctrl": "#1f77b4",
    "mnt": "#2ca02c",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one three-panel figure per protein: velocity jitter, speed jitter, "
            "and speed-vs-velocity scatter. Also save a Top10 subset to a separate folder."
        )
    )
    parser.add_argument("--patient-features-path", type=Path, default=DEFAULT_PATIENT_FEATURES_PATH)
    parser.add_argument("--ranking-path", type=Path, default=DEFAULT_RANKING_PATH)
    parser.add_argument("--output-base-dir", type=Path, default=DEFAULT_OUTPUT_BASE_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def load_inputs(patient_features_path: Path, ranking_path: Path):
    patient_df = pd.read_csv(patient_features_path)
    ranking_df = pd.read_csv(ranking_path)
    patient_df["group_short"] = patient_df["group_short"].astype(str)
    return patient_df, ranking_df


def build_protein_ranking(ranking_df: pd.DataFrame):
    pivot = (
        ranking_df.pivot_table(
            index="protein_id",
            columns="feature_type",
            values="mean_abs_value",
            aggfunc="first",
        )
        .rename_axis(columns=None)
        .reset_index()
    )
    if "speed" not in pivot.columns:
        pivot["speed"] = 0.0
    if "velocity" not in pivot.columns:
        pivot["velocity"] = 0.0

    pivot["speed"] = pivot["speed"].fillna(0.0)
    pivot["velocity"] = pivot["velocity"].fillna(0.0)
    pivot["combined_score"] = pivot["speed"] + pivot["velocity"]
    pivot["max_component_score"] = pivot[["speed", "velocity"]].max(axis=1)
    pivot = pivot.sort_values(
        ["combined_score", "max_component_score", "protein_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    pivot.insert(0, "protein_rank", np.arange(1, len(pivot) + 1))
    return pivot


def add_group_jitter(ax, df: pd.DataFrame, feature_col: str, ylabel: str):
    rng = np.random.default_rng(0)
    for idx, group_short in enumerate(GROUP_ORDER):
        group_df = df.loc[df["group_short"] == group_short].copy()
        if group_df.empty:
            continue
        jitter = rng.uniform(-0.14, 0.14, size=len(group_df))
        x = np.full(len(group_df), idx, dtype=float) + jitter
        ax.scatter(
            x,
            group_df[feature_col],
            s=48,
            alpha=0.82,
            color=GROUP_PALETTE[group_short],
            edgecolors="white",
            linewidths=0.6,
            zorder=2,
        )
        mean_val = float(group_df[feature_col].mean())
        ax.hlines(mean_val, idx - 0.22, idx + 0.22, color="black", linewidth=1.8, zorder=3)

    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_xlabel("Group")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)


def add_speed_velocity_scatter(ax, df: pd.DataFrame, velocity_col: str, speed_col: str):
    for group_short in GROUP_ORDER:
        group_df = df.loc[df["group_short"] == group_short].copy()
        if group_df.empty:
            continue
        ax.scatter(
            group_df[velocity_col],
            group_df[speed_col],
            s=52,
            alpha=0.82,
            color=GROUP_PALETTE[group_short],
            edgecolors="white",
            linewidths=0.6,
            label=group_short,
        )

    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Speed")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best")


def plot_protein_triptych(patient_df: pd.DataFrame, protein_row: pd.Series, output_path: Path):
    protein_id = protein_row["protein_id"]
    velocity_col = f"{protein_id}__velocity"
    speed_col = f"{protein_id}__speed"

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    add_group_jitter(axes[0], patient_df, velocity_col, ylabel="Velocity")
    axes[0].set_title(f"{protein_id} velocity")

    add_group_jitter(axes[1], patient_df, speed_col, ylabel="Speed")
    axes[1].set_title(f"{protein_id} speed")

    add_speed_velocity_scatter(axes[2], patient_df, velocity_col, speed_col)
    axes[2].set_title(f"{protein_id} speed vs velocity")

    fig.suptitle(
        f"{protein_id} | rank={int(protein_row['protein_rank'])} | "
        f"combined={protein_row['combined_score']:.4f}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.82, wspace=0.28)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    patient_df, ranking_df = load_inputs(args.patient_features_path, args.ranking_path)
    protein_ranking = build_protein_ranking(ranking_df)

    all_dir = args.output_base_dir / "protein_triptychs_all"
    top_dir = args.output_base_dir / f"protein_triptychs_top{args.top_k}"
    all_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for _, row in protein_ranking.iterrows():
        protein_id = row["protein_id"]
        velocity_col = f"{protein_id}__velocity"
        speed_col = f"{protein_id}__speed"
        if velocity_col not in patient_df.columns or speed_col not in patient_df.columns:
            continue

        output_path = all_dir / f"{protein_id}_speed_velocity_triptych.png"
        plot_protein_triptych(patient_df, row, output_path)

        manifest_rows.append(
            {
                "protein_rank": int(row["protein_rank"]),
                "protein_id": protein_id,
                "combined_score": float(row["combined_score"]),
                "speed_mean_abs_value": float(row["speed"]),
                "velocity_mean_abs_value": float(row["velocity"]),
                "all_plot_path": str(output_path),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = args.output_base_dir / "protein_triptych_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    top_df = manifest_df.head(args.top_k).copy()
    for _, row in top_df.iterrows():
        src = Path(row["all_plot_path"])
        dst = top_dir / src.name
        shutil.copy2(src, dst)

    top_manifest_path = args.output_base_dir / f"protein_triptych_top{args.top_k}.csv"
    top_df.to_csv(top_manifest_path, index=False)

    print(f"Saved all protein triptychs to {all_dir}")
    print(f"Saved Top{args.top_k} protein triptychs to {top_dir}")
    print(f"Saved manifest to {manifest_path}")
    print(f"Saved Top{args.top_k} manifest to {top_manifest_path}")


if __name__ == "__main__":
    main()
