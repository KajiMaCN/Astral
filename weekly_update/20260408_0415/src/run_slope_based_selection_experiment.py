import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METADATA_PATH = Path("/workspace/Data/old/metadata-fep_delta.csv")
DEFAULT_MATRIX_PATH = Path("/workspace/Data/20260401_0408/cmc-combat_0409.csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/weekly_update/20260408_0415/temp")
TARGET_GROUPS = ["Convert", "Healthy control", "Maintain"]
GROUP_LABEL_MAP = {
    "Convert": "cvt",
    "Healthy control": "ctrl",
    "Maintain": "mnt",
}
GROUP_PALETTE = {
    "cvt": "#d62728",
    "ctrl": "#1f77b4",
    "mnt": "#2ca02c",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Strictly follow the slope-feature definition from the meeting notes: "
            "build patient-level velocity and speed features from longitudinal protein trajectories."
        )
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--jitter-feature",
        type=str,
        default=None,
        help=(
            "Optional feature to plot. Use average_velocity_of_change, "
            "average_speed_of_change, or a feature like P13645__velocity."
        ),
    )
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    sample_col = metadata.columns[0]
    metadata = metadata.rename(columns={sample_col: "sample_id"})
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["sn"] = metadata["sn"].astype(str)
    metadata["timepoint"] = pd.to_numeric(metadata["timepoint"], errors="coerce")
    metadata["group"] = metadata["group"].astype(str)
    metadata["state"] = metadata["state"].fillna("Unknown").astype(str)
    metadata = metadata.loc[metadata["group"].isin(TARGET_GROUPS)].copy()
    return metadata


def load_matrix(matrix_path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(matrix_path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix


def build_dataset(metadata: pd.DataFrame, matrix: pd.DataFrame):
    covered = metadata.loc[metadata["sample_id"].isin(matrix.columns)].copy()
    covered = covered.sort_values(["group", "sn", "timepoint"]).reset_index(drop=True)
    protein_matrix = matrix.loc[:, covered["sample_id"]].T

    if protein_matrix.isna().any().any():
        missing_counts = protein_matrix.isna().sum(axis=0)
        bad_features = missing_counts.loc[missing_counts > 0].index.tolist()
        raise ValueError(
            "Protein matrix contains missing values for the selected samples. "
            f"Examples: {bad_features[:10]}"
        )

    subject_timepoint_counts = covered.groupby("sn").size()
    valid_subjects = subject_timepoint_counts.loc[subject_timepoint_counts >= 2].index.tolist()
    covered = covered.loc[covered["sn"].isin(valid_subjects)].copy()
    protein_matrix = protein_matrix.loc[covered["sample_id"]]
    return covered, protein_matrix


def compute_velocity_and_speed(timepoints: np.ndarray, values: np.ndarray):
    duration = float(timepoints[-1] - timepoints[0])
    if duration <= 0:
        return {
            "velocity": np.nan,
            "speed": np.nan,
            "duration": duration,
        }

    velocity = float((values[-1] - values[0]) / duration)
    speed = float(np.abs(np.diff(values)).sum() / duration)
    return {
        "velocity": velocity,
        "speed": speed,
        "duration": duration,
    }


def build_feature_tables(metadata: pd.DataFrame, protein_matrix: pd.DataFrame):
    patient_rows = []
    patient_summary_rows = []
    feature_long_rows = []

    for sn, subject_meta in metadata.groupby("sn", sort=True):
        subject_meta = subject_meta.sort_values("timepoint").reset_index(drop=True)
        group = subject_meta["group"].iloc[0]
        group_short = GROUP_LABEL_MAP[group]
        sample_ids = subject_meta["sample_id"].tolist()
        timepoints = subject_meta["timepoint"].to_numpy(dtype=float)
        states = subject_meta["state"].tolist()
        subject_values = protein_matrix.loc[sample_ids]

        patient_row = {
            "sn": sn,
            "group": group,
            "group_short": group_short,
            "n_timepoints": int(len(timepoints)),
            "timepoints": ";".join(str(int(tp)) if float(tp).is_integer() else str(tp) for tp in timepoints),
            "states": ";".join(states),
        }

        velocity_values = []
        speed_values = []

        for protein_id in protein_matrix.columns:
            values = subject_values[protein_id].to_numpy(dtype=float)
            feature_values = compute_velocity_and_speed(timepoints=timepoints, values=values)

            velocity_feature_id = f"{protein_id}__velocity"
            speed_feature_id = f"{protein_id}__speed"
            patient_row[velocity_feature_id] = feature_values["velocity"]
            patient_row[speed_feature_id] = feature_values["speed"]

            velocity_values.append(feature_values["velocity"])
            speed_values.append(feature_values["speed"])

            feature_long_rows.append(
                {
                    "sn": sn,
                    "group": group,
                    "group_short": group_short,
                    "protein_id": protein_id,
                    "n_timepoints": int(len(timepoints)),
                    "timepoints": patient_row["timepoints"],
                    "states": patient_row["states"],
                    "total_duration": feature_values["duration"],
                    "velocity_feature_id": velocity_feature_id,
                    "velocity_value": feature_values["velocity"],
                    "speed_feature_id": speed_feature_id,
                    "speed_value": feature_values["speed"],
                }
            )

        patient_row["average_velocity_of_change"] = float(np.nanmean(velocity_values))
        patient_row["average_speed_of_change"] = float(np.nanmean(speed_values))
        patient_rows.append(patient_row)

        patient_summary_rows.append(
            {
                "sn": sn,
                "group": group,
                "group_short": group_short,
                "n_timepoints": int(len(timepoints)),
                "timepoints": patient_row["timepoints"],
                "states": patient_row["states"],
                "average_velocity_of_change": float(np.nanmean(velocity_values)),
                "average_speed_of_change": float(np.nanmean(speed_values)),
            }
        )

    patient_feature_df = pd.DataFrame(patient_rows)
    patient_summary_df = pd.DataFrame(patient_summary_rows)
    feature_long_df = pd.DataFrame(feature_long_rows)
    return patient_feature_df, patient_summary_df, feature_long_df


def build_feature_ranking(feature_long_df: pd.DataFrame):
    ranking_rows = []

    velocity_summary = (
        feature_long_df.groupby("protein_id", sort=True)
        .agg(
            mean_value=("velocity_value", "mean"),
            mean_abs_value=("velocity_value", lambda s: float(np.abs(s).mean())),
            std_value=("velocity_value", lambda s: float(np.std(s, ddof=0))),
        )
        .reset_index()
    )
    velocity_group_means = (
        feature_long_df.pivot_table(
            index="protein_id",
            columns="group_short",
            values="velocity_value",
            aggfunc="mean",
        )
        .reset_index()
    )
    velocity_summary = velocity_summary.merge(velocity_group_means, on="protein_id", how="left")
    for _, row in velocity_summary.iterrows():
        ranking_rows.append(
            {
                "feature_id": f"{row['protein_id']}__velocity",
                "protein_id": row["protein_id"],
                "feature_type": "velocity",
                "mean_value": float(row["mean_value"]),
                "mean_abs_value": float(row["mean_abs_value"]),
                "std_value": float(row["std_value"]),
                "cvt_mean_value": float(row.get("cvt", np.nan)),
                "ctrl_mean_value": float(row.get("ctrl", np.nan)),
                "mnt_mean_value": float(row.get("mnt", np.nan)),
            }
        )

    speed_summary = (
        feature_long_df.groupby("protein_id", sort=True)
        .agg(
            mean_value=("speed_value", "mean"),
            mean_abs_value=("speed_value", lambda s: float(np.abs(s).mean())),
            std_value=("speed_value", lambda s: float(np.std(s, ddof=0))),
        )
        .reset_index()
    )
    speed_group_means = (
        feature_long_df.pivot_table(
            index="protein_id",
            columns="group_short",
            values="speed_value",
            aggfunc="mean",
        )
        .reset_index()
    )
    speed_summary = speed_summary.merge(speed_group_means, on="protein_id", how="left")
    for _, row in speed_summary.iterrows():
        ranking_rows.append(
            {
                "feature_id": f"{row['protein_id']}__speed",
                "protein_id": row["protein_id"],
                "feature_type": "speed",
                "mean_value": float(row["mean_value"]),
                "mean_abs_value": float(row["mean_abs_value"]),
                "std_value": float(row["std_value"]),
                "cvt_mean_value": float(row.get("cvt", np.nan)),
                "ctrl_mean_value": float(row.get("ctrl", np.nan)),
                "mnt_mean_value": float(row.get("mnt", np.nan)),
            }
        )

    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["mean_abs_value", "std_value", "feature_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return ranking


def save_jitter_plot(plot_df: pd.DataFrame, feature_col: str, output_path: Path):
    if feature_col not in plot_df.columns:
        raise ValueError(f"Requested jitter feature not found: {feature_col}")

    order = ["cvt", "ctrl", "mnt"]
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    rng = np.random.default_rng(0)

    for idx, group_short in enumerate(order):
        group_df = plot_df.loc[plot_df["group_short"] == group_short].copy()
        if group_df.empty:
            continue
        jitter = rng.uniform(-0.13, 0.13, size=len(group_df))
        x = np.full(len(group_df), idx, dtype=float) + jitter
        ax.scatter(
            x,
            group_df[feature_col],
            s=70,
            alpha=0.85,
            color=GROUP_PALETTE[group_short],
            edgecolors="white",
            linewidths=0.8,
            label=group_short,
        )
        mean_val = group_df[feature_col].mean()
        ax.hlines(mean_val, idx - 0.22, idx + 0.22, color="black", linewidth=2.0)

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Group")
    ax.set_ylabel(feature_col)
    ax.set_title(f"Jitter plot for {feature_col}")
    ax.grid(axis="y", alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_path)
    protein_matrix = load_matrix(args.matrix_path)
    metadata, protein_matrix = build_dataset(metadata, protein_matrix)

    patient_feature_df, patient_summary_df, feature_long_df = build_feature_tables(
        metadata=metadata,
        protein_matrix=protein_matrix,
    )
    feature_ranking = build_feature_ranking(feature_long_df)

    jitter_feature = args.jitter_feature or "average_velocity_of_change"

    if jitter_feature in patient_summary_df.columns:
        jitter_source_df = patient_summary_df
    elif jitter_feature in patient_feature_df.columns:
        jitter_source_df = patient_feature_df
    else:
        raise ValueError(
            f"Requested jitter feature not found: {jitter_feature}. "
            "Use average_velocity_of_change, average_speed_of_change, or a patient-by-feature column name."
        )

    jitter_plot_path = args.output_dir / f"jitter_{jitter_feature}.png"
    save_jitter_plot(plot_df=jitter_source_df, feature_col=jitter_feature, output_path=jitter_plot_path)

    average_velocity_plot_path = args.output_dir / "jitter_average_velocity_of_change.png"
    average_speed_plot_path = args.output_dir / "jitter_average_speed_of_change.png"
    save_jitter_plot(
        plot_df=patient_summary_df,
        feature_col="average_velocity_of_change",
        output_path=average_velocity_plot_path,
    )
    save_jitter_plot(
        plot_df=patient_summary_df,
        feature_col="average_speed_of_change",
        output_path=average_speed_plot_path,
    )

    summary = {
        "metadata_path": str(args.metadata_path),
        "matrix_path": str(args.matrix_path),
        "sample_count": int(len(metadata)),
        "subject_count": int(patient_summary_df["sn"].nunique()),
        "group_counts": {
            group_short: int((patient_summary_df["group_short"] == group_short).sum())
            for group_short in ["cvt", "ctrl", "mnt"]
        },
        "protein_feature_count": int(protein_matrix.shape[1]),
        "slope_feature_count": int((protein_matrix.shape[1] * 2) + 2),
        "top_ranked_feature": str(feature_ranking.iloc[0]["feature_id"]),
        "jitter_feature": str(jitter_feature),
        "average_velocity_summary": {
            group_short: float(
                patient_summary_df.loc[
                    patient_summary_df["group_short"] == group_short, "average_velocity_of_change"
                ].mean()
            )
            for group_short in ["cvt", "ctrl", "mnt"]
        },
        "average_speed_summary": {
            group_short: float(
                patient_summary_df.loc[
                    patient_summary_df["group_short"] == group_short, "average_speed_of_change"
                ].mean()
            )
            for group_short in ["cvt", "ctrl", "mnt"]
        },
    }

    summary_path = args.output_dir / "slope_based_selection_summary.json"
    patient_summary_path = args.output_dir / "patient_slope_summary_features.csv"
    patient_feature_matrix_path = args.output_dir / "patient_by_slope_features.csv"
    feature_ranking_path = args.output_dir / "slope_feature_ranking.csv"
    feature_long_path = args.output_dir / "patient_protein_slope_features_long.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    patient_summary_df.to_csv(patient_summary_path, index=False)
    patient_feature_df.to_csv(patient_feature_matrix_path, index=False)
    feature_ranking.to_csv(feature_ranking_path, index=False)
    feature_long_df.to_csv(feature_long_path, index=False)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved patient summary features to {patient_summary_path}")
    print(f"Saved patient-by-feature matrix to {patient_feature_matrix_path}")
    print(f"Saved feature ranking to {feature_ranking_path}")
    print(f"Saved long table to {feature_long_path}")
    print(f"Saved jitter plots to {average_velocity_plot_path} and {average_velocity_plot_path.with_suffix('.pdf')}")
    print(f"Saved jitter plots to {average_speed_plot_path} and {average_speed_plot_path.with_suffix('.pdf')}")
    print(f"Saved requested jitter plot to {jitter_plot_path} and {jitter_plot_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
