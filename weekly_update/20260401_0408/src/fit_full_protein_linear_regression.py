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
import statsmodels.api as sm


DEFAULT_METADATA_PATH = Path("/workspace/Data/old/metadata-fep_delta.csv")
DEFAULT_MATRIX_PATH = Path("/workspace/Data/20260401_0408/cmc-combat_0409.csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/weekly_update/20260401_0408/temp")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit subject-specific protein ~ timepoint linear models for the "
            "longitudinal samples and summarize slope p-values."
        )
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    sample_col = metadata.columns[0]
    metadata = metadata.rename(columns={sample_col: "sample_id"})
    metadata["fep_delta"] = pd.to_numeric(metadata["fep_delta"], errors="coerce")
    metadata["timepoint"] = pd.to_numeric(metadata["timepoint"], errors="coerce")
    metadata = metadata.loc[metadata["fep_delta"].notna()].copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["sn"] = metadata["sn"].astype(str)
    metadata["state"] = metadata["state"].fillna("Unknown").astype(str)
    return metadata


def load_matrix(matrix_path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(matrix_path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix


def build_dataset(metadata: pd.DataFrame, matrix: pd.DataFrame):
    sample_ids = [sample_id for sample_id in metadata["sample_id"] if sample_id in matrix.columns]
    missing_samples = sorted(set(metadata["sample_id"]) - set(sample_ids))
    if missing_samples:
        raise ValueError(
            "Some metadata samples were not found in the protein matrix: "
            + ", ".join(missing_samples[:10])
        )

    metadata = metadata.set_index("sample_id").loc[sample_ids].reset_index()
    protein_matrix = matrix.loc[:, sample_ids].T

    if protein_matrix.isna().any().any():
        missing_counts = protein_matrix.isna().sum(axis=0)
        bad_features = missing_counts.loc[missing_counts > 0].index.tolist()
        raise ValueError(
            "Protein matrix contains missing values for the modeling subset. "
            f"Examples: {bad_features[:10]}"
        )

    return metadata, protein_matrix


def fit_subject_protein_model(timepoints: np.ndarray, values: np.ndarray):
    design_matrix = sm.add_constant(timepoints, has_constant="add")
    result = sm.OLS(values, design_matrix).fit()
    predicted = result.predict(design_matrix)

    ss_res = float(np.sum((values - predicted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    slope = float(result.params[1])
    intercept = float(result.params[0])
    slope_stderr = float(result.bse[1]) if len(result.bse) > 1 else np.nan
    slope_pvalue = float(result.pvalues[1]) if len(result.pvalues) > 1 else np.nan

    return {
        "intercept": intercept,
        "slope": slope,
        "predicted": predicted,
        "r2": r2,
        "slope_stderr": slope_stderr,
        "slope_pvalue": slope_pvalue,
        "df_resid": float(result.df_resid),
    }


def build_subject_level_outputs(metadata: pd.DataFrame, protein_matrix: pd.DataFrame):
    coefficient_rows = []
    prediction_rows = []

    for sn, subject_meta in metadata.groupby("sn", sort=True):
        subject_meta = subject_meta.sort_values("timepoint").reset_index(drop=True)
        sample_ids = subject_meta["sample_id"].tolist()
        timepoints = subject_meta["timepoint"].to_numpy(dtype=float)
        states = subject_meta["state"].tolist()
        subject_values = protein_matrix.loc[sample_ids]

        for protein_id in protein_matrix.columns:
            values = subject_values[protein_id].to_numpy(dtype=float)
            fit = fit_subject_protein_model(timepoints=timepoints, values=values)

            coefficient_rows.append(
                {
                    "sn": sn,
                    "protein_id": protein_id,
                    "n_timepoints": int(len(timepoints)),
                    "timepoints": ";".join(str(int(tp)) if float(tp).is_integer() else str(tp) for tp in timepoints),
                    "states": ";".join(states),
                    "intercept": fit["intercept"],
                    "slope": fit["slope"],
                    "slope_abs": abs(fit["slope"]),
                    "r2": fit["r2"],
                    "slope_stderr": fit["slope_stderr"],
                    "slope_pvalue": fit["slope_pvalue"],
                    "slope_pvalue_finite": bool(np.isfinite(fit["slope_pvalue"])),
                    "df_resid": fit["df_resid"],
                }
            )

            for sample_id, timepoint, state, observed, predicted in zip(
                sample_ids,
                timepoints,
                states,
                values,
                fit["predicted"],
            ):
                prediction_rows.append(
                    {
                        "sn": sn,
                        "sample_id": sample_id,
                        "timepoint": float(timepoint),
                        "state": state,
                        "protein_id": protein_id,
                        "observed_protein_value": float(observed),
                        "predicted_protein_value": float(predicted),
                        "residual": float(observed - predicted),
                        "slope": fit["slope"],
                        "slope_pvalue": fit["slope_pvalue"],
                    }
                )

    coefficients = pd.DataFrame(coefficient_rows)
    coefficients = coefficients.sort_values(
        ["slope_pvalue_finite", "slope_abs", "sn", "protein_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    predictions = pd.DataFrame(prediction_rows)
    return coefficients, predictions


def build_protein_summary(coefficients: pd.DataFrame):
    protein_summary = (
        coefficients.groupby("protein_id", sort=True)
        .agg(
            subject_fit_count=("sn", "count"),
            finite_pvalue_count=("slope_pvalue_finite", "sum"),
            significant_subject_count_p_lt_0_05=("slope_pvalue", lambda s: int((s < 0.05).fillna(False).sum())),
            significant_subject_count_p_lt_0_10=("slope_pvalue", lambda s: int((s < 0.10).fillna(False).sum())),
            mean_slope=("slope", "mean"),
            median_slope=("slope", "median"),
            mean_abs_slope=("slope_abs", "mean"),
        )
        .reset_index()
    )
    protein_summary = protein_summary.sort_values(
        [
            "significant_subject_count_p_lt_0_05",
            "finite_pvalue_count",
            "mean_abs_slope",
            "protein_id",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return protein_summary


def build_pvalue_tables(coefficients: pd.DataFrame):
    pvalue_long = coefficients[
        [
            "sn",
            "protein_id",
            "n_timepoints",
            "timepoints",
            "states",
            "slope",
            "slope_pvalue",
            "slope_pvalue_finite",
            "df_resid",
        ]
    ].copy()
    pvalue_long = pvalue_long.sort_values(
        ["slope_pvalue_finite", "slope_pvalue", "sn", "protein_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    pvalue_matrix = coefficients.pivot(index="protein_id", columns="sn", values="slope_pvalue")
    pvalue_matrix = pvalue_matrix.sort_index().reset_index()
    return pvalue_long, pvalue_matrix


def save_summary_plot(coefficients: pd.DataFrame, protein_summary: pd.DataFrame, summary: dict, output_path: Path):
    timepoint_fit_summary = (
        coefficients.groupby("n_timepoints")
        .agg(
            total_fits=("protein_id", "count"),
            finite_pvalue_fits=("slope_pvalue_finite", "sum"),
        )
        .reset_index()
        .sort_values("n_timepoints")
    )

    fig = plt.figure(figsize=(12.8, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.65], wspace=0.28)

    ax_left = fig.add_subplot(gs[0, 0])
    x = np.arange(len(timepoint_fit_summary))
    ax_left.bar(x, timepoint_fit_summary["total_fits"], color="#c7d4ea", label="All fits")
    ax_left.bar(x, timepoint_fit_summary["finite_pvalue_fits"], color="#1f77b4", label="Finite p-values")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(timepoint_fit_summary["n_timepoints"].astype(str).tolist())
    ax_left.set_xlabel("Number of timepoints in a subject fit")
    ax_left.set_ylabel("Subject-protein fits")
    ax_left.set_title("P-value availability by longitudinal fit size")
    ax_left.legend(frameon=False)
    ax_left.grid(axis="y", alpha=0.2)

    ax_right = fig.add_subplot(gs[0, 1])
    top_df = protein_summary.head(15).iloc[::-1]
    ax_right.barh(top_df["protein_id"], top_df["significant_subject_count_p_lt_0_05"], color="#d62728")
    ax_right.set_xlabel("Subjects with slope p < 0.05")
    ax_right.set_title("Top proteins by subject-level significant slopes")
    ax_right.grid(axis="x", alpha=0.2)

    annotation = (
        f"Subjects: {summary['subject_count']}\n"
        f"Proteins: {summary['protein_feature_count']}\n"
        f"Total subject-protein fits: {summary['total_subject_protein_fits']}\n"
        f"Finite slope p-values: {summary['finite_pvalue_fit_count']}\n"
        f"Subjects with only 2 timepoints: {summary['subject_count_with_2_timepoints']}\n"
        f"Use p-values for feature selection: {summary['can_use_coefficient_pvalues_for_feature_selection']}"
    )
    fig.text(0.5, 0.02, annotation, ha="center", va="bottom", fontsize=10, family="monospace")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.16)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_path)
    protein_matrix = load_matrix(args.matrix_path)
    metadata, protein_matrix = build_dataset(metadata, protein_matrix)

    coefficients, predictions = build_subject_level_outputs(metadata=metadata, protein_matrix=protein_matrix)
    protein_summary = build_protein_summary(coefficients)
    pvalue_long, pvalue_matrix = build_pvalue_tables(coefficients)

    subject_timepoint_counts = metadata.groupby("sn").size()
    finite_pvalue_mask = coefficients["slope_pvalue_finite"]

    can_use_pvalues = False
    pvalue_assessment_reason = (
        "These p-values come from separate subject-specific protein ~ timepoint fits, not from one global "
        "per-protein model. Also, many subjects have only 2 timepoints, which leaves no residual degrees "
        "of freedom, and most remaining fits have only 3 timepoints, so the slope p-values are sparse and unstable."
    )

    summary = {
        "metadata_path": str(args.metadata_path),
        "matrix_path": str(args.matrix_path),
        "sample_count": int(len(metadata)),
        "subject_count": int(metadata["sn"].nunique()),
        "protein_feature_count": int(protein_matrix.shape[1]),
        "total_subject_protein_fits": int(len(coefficients)),
        "timepoint_count_distribution_by_subject": {
            str(int(k)): int(v) for k, v in subject_timepoint_counts.value_counts().sort_index().items()
        },
        "subject_count_with_2_timepoints": int((subject_timepoint_counts == 2).sum()),
        "subject_count_with_3_or_more_timepoints": int((subject_timepoint_counts >= 3).sum()),
        "finite_pvalue_fit_count": int(finite_pvalue_mask.sum()),
        "nonfinite_pvalue_fit_count": int((~finite_pvalue_mask).sum()),
        "protein_count_with_any_finite_pvalue_fit": int(protein_summary["finite_pvalue_count"].gt(0).sum()),
        "protein_count_with_any_subject_p_lt_0_05": int(
            protein_summary["significant_subject_count_p_lt_0_05"].gt(0).sum()
        ),
        "protein_count_with_any_subject_p_lt_0_10": int(
            protein_summary["significant_subject_count_p_lt_0_10"].gt(0).sum()
        ),
        "can_use_coefficient_pvalues_for_feature_selection": can_use_pvalues,
        "pvalue_assessment_reason": pvalue_assessment_reason,
        "top_proteins_by_significant_subject_count": protein_summary.head(20)["protein_id"].tolist(),
    }

    summary_path = args.output_dir / "linear_regression_timepoint_by_subject_summary.json"
    coefficients_path = args.output_dir / "linear_regression_timepoint_by_subject_coefficients.csv"
    predictions_path = args.output_dir / "linear_regression_timepoint_by_subject_predictions.csv"
    protein_summary_path = args.output_dir / "linear_regression_timepoint_by_subject_protein_summary.csv"
    pvalue_long_path = args.output_dir / "linear_regression_timepoint_by_subject_pvalues.csv"
    pvalue_matrix_path = args.output_dir / "linear_regression_timepoint_by_subject_pvalue_matrix.csv"
    plot_path = args.output_dir / "linear_regression_timepoint_by_subject_summary_plot.png"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    coefficients.to_csv(coefficients_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    protein_summary.to_csv(protein_summary_path, index=False)
    pvalue_long.to_csv(pvalue_long_path, index=False)
    pvalue_matrix.to_csv(pvalue_matrix_path, index=False)
    save_summary_plot(
        coefficients=coefficients,
        protein_summary=protein_summary,
        summary=summary,
        output_path=plot_path,
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved coefficient table to {coefficients_path}")
    print(f"Saved prediction table to {predictions_path}")
    print(f"Saved protein summary table to {protein_summary_path}")
    print(f"Saved p-value table to {pvalue_long_path}")
    print(f"Saved p-value matrix to {pvalue_matrix_path}")
    print(f"Saved plots to {plot_path} and {plot_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
