import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA


DEFAULT_METADATA_PATH = Path("/workspace/Data/old/metadata-fep_delta.csv")
DEFAULT_MATRIX_PATH = Path("/workspace/Data/20260401_0408/cmc-combat_0409.csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/weekly_update/20260408_0415/temp")
DEFAULT_OUTPUT_SUBDIR = "cca_m2c_train_convert_transform_all"

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
STATE_MARKERS = {
    "UHR": "o",
    "FEP": "s",
    "Unknown": "D",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the CCA experiment described in the meeting notes: find the protein "
            "linear combination that correlates most with month_of_conversion, then "
            "project cvt/ctrl/mnt samples onto CCA1."
        )
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-subdir", type=str, default=DEFAULT_OUTPUT_SUBDIR)
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    sample_col = metadata.columns[0]
    metadata = metadata.rename(columns={sample_col: "sample_id"})
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["sn"] = metadata["sn"].astype(str)
    metadata["group"] = metadata["group"].astype(str)
    metadata["state"] = metadata["state"].fillna("Unknown").astype(str)
    metadata["timepoint"] = pd.to_numeric(metadata["timepoint"], errors="coerce")
    metadata["month_of_conversion"] = pd.to_numeric(metadata["month_of_conversion"], errors="coerce")
    metadata["fep_delta"] = pd.to_numeric(metadata["fep_delta"], errors="coerce")
    metadata = metadata.loc[metadata["group"].isin(TARGET_GROUPS)].copy()
    metadata["group_short"] = metadata["group"].map(GROUP_LABEL_MAP)
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

    return covered, protein_matrix


def fit_cca(convert_meta: pd.DataFrame, convert_x: pd.DataFrame):
    y = convert_meta["fep_delta"].to_numpy(dtype=float).reshape(-1, 1)
    cca = CCA(n_components=1, scale=True, max_iter=1000)
    cca.fit(convert_x.to_numpy(dtype=float), y)
    x_scores, y_scores = cca.transform(convert_x.to_numpy(dtype=float), y)

    orientation = 1.0
    raw_corr = np.corrcoef(x_scores[:, 0], y[:, 0])[0, 1]
    if np.isfinite(raw_corr) and raw_corr < 0:
        orientation = -1.0
        x_scores = x_scores * orientation
        y_scores = y_scores * orientation

    result = {
        "cca": cca,
        "orientation": orientation,
        "x_scores": x_scores[:, 0],
        "y_scores": y_scores[:, 0],
        "raw_y": y[:, 0],
        "canonical_correlation": float(np.corrcoef(x_scores[:, 0], y_scores[:, 0])[0, 1]),
        "cca1_m2c_correlation": float(np.corrcoef(x_scores[:, 0], y[:, 0])[0, 1]),
    }
    return result


def transform_all_samples(cca_result: dict, all_meta: pd.DataFrame, all_x: pd.DataFrame):
    cca = cca_result["cca"]
    orientation = cca_result["orientation"]
    transformed = cca.transform(all_x.to_numpy(dtype=float))
    if transformed.ndim == 2:
        transformed = transformed[:, 0]
    transformed = transformed * orientation

    sample_scores = all_meta.copy()
    sample_scores["cca1_score"] = transformed
    sample_scores["is_convert_fit_sample"] = sample_scores["group"].eq("Convert")
    return sample_scores


def build_convert_fit_scores(convert_meta: pd.DataFrame, cca_result: dict):
    fit_scores = convert_meta.copy()
    fit_scores["cca1_score"] = cca_result["x_scores"]
    fit_scores["cca1_target_score"] = cca_result["y_scores"]
    fit_scores["cca1_target_raw_m2c"] = cca_result["raw_y"]
    fit_scores["fep_delta"] = pd.to_numeric(fit_scores["fep_delta"], errors="coerce")
    return fit_scores


def build_feature_weights(protein_ids: pd.Index, cca_result: dict):
    cca = cca_result["cca"]
    orientation = cca_result["orientation"]
    weights = pd.DataFrame(
        {
            "protein_id": protein_ids.to_numpy(),
            "cca1_weight": cca.x_weights_[:, 0] * orientation,
            "cca1_loading": cca.x_loadings_[:, 0] * orientation,
            "cca1_rotation": cca.x_rotations_[:, 0] * orientation,
        }
    )
    weights["abs_cca1_weight"] = weights["cca1_weight"].abs()
    weights = weights.sort_values(
        ["abs_cca1_weight", "protein_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    weights.insert(0, "feature_rank", np.arange(1, len(weights) + 1))
    return weights


def build_subject_summary(sample_scores: pd.DataFrame):
    subject_summary = (
        sample_scores.groupby(["sn", "group", "group_short"], as_index=False)
        .agg(
            n_timepoints=("sample_id", "size"),
            mean_cca1_score=("cca1_score", "mean"),
            median_cca1_score=("cca1_score", "median"),
            min_cca1_score=("cca1_score", "min"),
            max_cca1_score=("cca1_score", "max"),
            mean_timepoint=("timepoint", "mean"),
        )
    )
    return subject_summary


def add_secondary_embedding_axis(cca_result: dict, all_x: pd.DataFrame, sample_scores: pd.DataFrame):
    cca = cca_result["cca"]
    x_mean = pd.Series(cca._x_mean, index=all_x.columns)
    x_std = pd.Series(cca._x_std, index=all_x.columns).replace(0.0, 1.0)
    x_scaled = (all_x - x_mean) / x_std

    cca1_direction = cca.x_rotations_[:, 0] * cca_result["orientation"]
    cca1_norm = np.linalg.norm(cca1_direction)
    if cca1_norm <= 0:
        raise ValueError("CCA1 direction has zero norm; cannot build a secondary embedding axis.")
    cca1_unit = cca1_direction / cca1_norm

    x_scaled_array = x_scaled.to_numpy(dtype=float)
    cca1_projection = x_scaled_array @ cca1_unit
    x_residual = x_scaled_array - np.outer(cca1_projection, cca1_unit)

    pca = PCA(n_components=1, random_state=0)
    axis2_scores = pca.fit_transform(x_residual)[:, 0]

    sample_scores = sample_scores.copy()
    sample_scores["embedding_axis2"] = axis2_scores

    axis2_summary = {
        "name": "residual_pc1",
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
    }
    return sample_scores, axis2_summary, pca


def save_group_jitter_plot(sample_scores: pd.DataFrame, output_path: Path):
    order = ["cvt", "ctrl", "mnt"]
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    rng = np.random.default_rng(0)

    for idx, group_short in enumerate(order):
        group_df = sample_scores.loc[sample_scores["group_short"] == group_short].copy()
        if group_df.empty:
            continue
        jitter = rng.uniform(-0.14, 0.14, size=len(group_df))
        x = np.full(len(group_df), idx, dtype=float) + jitter
        ax.scatter(
            x,
            group_df["cca1_score"],
            s=65,
            alpha=0.8,
            color=GROUP_PALETTE[group_short],
            edgecolors="white",
            linewidths=0.7,
            label=group_short,
        )
        mean_val = float(group_df["cca1_score"].mean())
        ax.hlines(mean_val, idx - 0.24, idx + 0.24, color="black", linewidth=2.0)

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Group")
    ax.set_ylabel("CCA1 score")
    ax.set_title("CCA1 jitter plot across cvt / ctrl / mnt")
    ax.grid(axis="y", alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_cca_embedding_plot(sample_scores: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    order = ["cvt", "ctrl", "mnt"]

    convert_df = sample_scores.loc[sample_scores["group_short"] == "cvt"].copy()
    for sn, subject_df in convert_df.groupby("sn", sort=True):
        subject_df = subject_df.sort_values("timepoint")
        if len(subject_df) > 1:
            ax.plot(
                subject_df["cca1_score"],
                subject_df["embedding_axis2"],
                color="#d8dce3",
                linewidth=1.0,
                alpha=0.85,
                zorder=1,
            )

    for group_short in order:
        group_df = sample_scores.loc[sample_scores["group_short"] == group_short].copy()
        if group_df.empty:
            continue
        ax.scatter(
            group_df["cca1_score"],
            group_df["embedding_axis2"],
            s=64,
            alpha=0.82,
            color=GROUP_PALETTE[group_short],
            edgecolors="white",
            linewidths=0.7,
            label=group_short,
            zorder=2,
        )

    ax.set_xlabel("CCA1")
    ax.set_ylabel("Axis 2 (residual PC1)")
    ax.set_title("CCA embedding plot")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_2d_cca_plot(convert_fit_scores: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    point_color = "#5aa857"
    line_color = "#cfd4dc"

    for sn, subject_df in convert_fit_scores.groupby("sn", sort=True):
        subject_df = subject_df.sort_values("timepoint")
        ax.plot(
            subject_df["cca1_score"],
            subject_df["fep_delta"],
            color=line_color,
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )

        for _, row in subject_df.iterrows():
            ax.scatter(
                row["cca1_score"],
                row["fep_delta"],
                s=78,
                color=point_color,
                alpha=0.92,
                edgecolors="#f7f7f7",
                linewidths=0.7,
                marker="o",
                zorder=2,
            )

    corr = np.corrcoef(
        convert_fit_scores["cca1_score"].to_numpy(dtype=float),
        convert_fit_scores["fep_delta"].to_numpy(dtype=float),
    )[0, 1]
    ax.set_xlabel("CCA1 score")
    ax.set_ylabel("fep_delta")
    ax.set_title(f"CCA1 trajectory plot for convert samples (r = {corr:.2f})")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_summary(
    all_meta: pd.DataFrame,
    fit_meta: pd.DataFrame,
    weights: pd.DataFrame,
    cca_result: dict,
    axis2_summary: dict,
):
    group_counts = all_meta["group_short"].value_counts().sort_index().to_dict()
    top_features = weights.head(10)[["protein_id", "cca1_weight"]].to_dict(orient="records")
    return {
        "metadata_path": str(DEFAULT_METADATA_PATH),
        "matrix_path": str(DEFAULT_MATRIX_PATH),
        "analysis": "cca_m2c_protein_linear_combination",
        "fit_group": "Convert",
        "projection_groups": ["cvt", "ctrl", "mnt"],
        "target": "month_of_conversion",
        "fit_sample_count": int(len(fit_meta)),
        "fit_subject_count": int(fit_meta["sn"].nunique()),
        "projection_sample_count": int(len(all_meta)),
        "projection_subject_count": int(all_meta["sn"].nunique()),
        "group_counts": {k: int(v) for k, v in group_counts.items()},
        "protein_feature_count": int(len(weights)),
        "n_components": 1,
        "canonical_correlation": float(cca_result["canonical_correlation"]),
        "cca1_m2c_correlation": float(cca_result["cca1_m2c_correlation"]),
        "secondary_embedding_axis": axis2_summary,
        "two_dimensional_plot_note": (
            "The trajectory plot uses "
            "fep_delta on the x-axis and CCA1 score on the y-axis, with each convert "
            "patient connected across time."
        ),
        "cca_embedding_plot_note": (
            "Because the target is one-dimensional, true CCA provides one supervised axis. "
            "The second plotting axis is residual PC1 from the protein space after removing CCA1."
        ),
        "top_cca1_features": top_features,
    }


def main():
    args = parse_args()
    output_dir = args.output_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_path)
    matrix = load_matrix(args.matrix_path)
    all_meta, all_x = build_dataset(metadata, matrix)

    fit_meta = all_meta.loc[all_meta["group"] == "Convert"].copy()
    fit_meta = fit_meta.loc[fit_meta["month_of_conversion"].notna()].copy()
    fit_x = all_x.loc[fit_meta["sample_id"]].copy()

    cca_result = fit_cca(fit_meta, fit_x)
    sample_scores = transform_all_samples(cca_result, all_meta, all_x)
    sample_scores, axis2_summary, axis2_pca = add_secondary_embedding_axis(cca_result, all_x, sample_scores)
    convert_fit_scores = build_convert_fit_scores(fit_meta, cca_result)
    convert_fit_scores = convert_fit_scores.merge(
        sample_scores.loc[:, ["sample_id", "embedding_axis2"]],
        on="sample_id",
        how="left",
    )
    feature_weights = build_feature_weights(all_x.columns, cca_result)
    subject_summary = build_subject_summary(sample_scores)
    summary = build_summary(all_meta, fit_meta, feature_weights, cca_result, axis2_summary)

    sample_scores_path = output_dir / "cca_m2c_sample_scores.csv"
    convert_scores_path = output_dir / "cca_m2c_convert_fit_scores.csv"
    subject_summary_path = output_dir / "cca_m2c_subject_summary.csv"
    feature_weights_path = output_dir / "cca_m2c_feature_weights.csv"
    top_features_path = output_dir / "cca_m2c_top_features.csv"
    summary_path = output_dir / "cca_m2c_summary.json"
    model_bundle_path = output_dir / "cca_m2c_model_bundle.plk"
    jitter_path = output_dir / "cca_m2c_cca1_jitter_by_group.png"
    plot2d_path = output_dir / "cca_m2c_2d_plot.png"
    embedding_plot_path = output_dir / "cca_m2c_embedding_plot.png"

    sample_scores.to_csv(sample_scores_path, index=False)
    convert_fit_scores.to_csv(convert_scores_path, index=False)
    subject_summary.to_csv(subject_summary_path, index=False)
    feature_weights.to_csv(feature_weights_path, index=False)
    feature_weights.head(30).to_csv(top_features_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with model_bundle_path.open("wb") as f:
        pickle.dump(
            {
                "cca_model": cca_result["cca"],
                "orientation": cca_result["orientation"],
                "feature_names": all_x.columns.tolist(),
                "fit_group": "Convert",
                "target_name": "fep_delta",
                "output_subdir": args.output_subdir,
                "secondary_embedding_pca": axis2_pca,
                "secondary_embedding_axis_name": axis2_summary["name"],
            },
            f,
        )

    save_group_jitter_plot(sample_scores, jitter_path)
    save_2d_cca_plot(convert_fit_scores, plot2d_path)
    save_cca_embedding_plot(sample_scores, embedding_plot_path)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved sample scores to {sample_scores_path}")
    print(f"Saved convert fit scores to {convert_scores_path}")
    print(f"Saved subject summary to {subject_summary_path}")
    print(f"Saved feature weights to {feature_weights_path}")
    print(f"Saved top features to {top_features_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved model bundle to {model_bundle_path}")
    print(f"Saved jitter plot to {jitter_path} and {jitter_path.with_suffix('.pdf')}")
    print(f"Saved 2D plot to {plot2d_path} and {plot2d_path.with_suffix('.pdf')}")
    print(f"Saved embedding plot to {embedding_plot_path} and {embedding_plot_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
