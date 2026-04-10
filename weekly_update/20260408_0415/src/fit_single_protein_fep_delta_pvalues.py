import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


DEFAULT_METADATA_PATH = Path("/workspace/Data/old/metadata-fep_delta.csv")
DEFAULT_MATRIX_PATH = Path("/workspace/Data/20260401_0408/cmc-combat_0409.csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/weekly_update/20260408_0415/temp")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Redo the protein-level p-value analysis from the meeting notes by "
            "fitting one linear model per protein: fep_delta ~ protein."
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
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["fep_delta"] = pd.to_numeric(metadata["fep_delta"], errors="coerce")
    metadata["timepoint"] = pd.to_numeric(metadata["timepoint"], errors="coerce")
    metadata["month_of_conversion"] = pd.to_numeric(metadata["month_of_conversion"], errors="coerce")
    metadata = metadata.loc[metadata["fep_delta"].notna()].copy()
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

    aligned_metadata = metadata.set_index("sample_id").loc[sample_ids].reset_index()
    x = matrix.loc[:, sample_ids].T

    if x.isna().any().any():
        missing_counts = x.isna().sum(axis=0)
        bad_features = missing_counts.loc[missing_counts > 0].index.tolist()
        raise ValueError(
            "Protein matrix contains missing values for the analysis subset. "
            f"Examples: {bad_features[:10]}"
        )

    y = aligned_metadata["fep_delta"].astype(float)
    return aligned_metadata, x, y


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denom <= 0:
        return np.nan
    return float(np.sum(x_centered * y_centered) / denom)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return safe_pearson(x_rank, y_rank)


def fit_one_protein(protein_id: str, protein_values: pd.Series, y: pd.Series) -> dict:
    protein_array = protein_values.to_numpy(dtype=float)
    y_array = y.to_numpy(dtype=float)
    variance = float(np.var(protein_array, ddof=0))

    if np.isclose(variance, 0.0, atol=1e-15):
        return {
            "protein_id": protein_id,
            "sample_count": int(len(y)),
            "protein_variance": variance,
            "intercept": np.nan,
            "slope_coef": np.nan,
            "intercept_pvalue": np.nan,
            "slope_pvalue": np.nan,
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "status": "zero_variance",
        }

    design = sm.add_constant(pd.DataFrame({"protein_value": protein_array}), has_constant="add")
    result = sm.OLS(y_array, design).fit()

    return {
        "protein_id": protein_id,
        "sample_count": int(len(y)),
        "protein_variance": variance,
        "intercept": float(result.params["const"]),
        "slope_coef": float(result.params["protein_value"]),
        "intercept_pvalue": float(result.pvalues["const"]),
        "slope_pvalue": float(result.pvalues["protein_value"]),
        "r_squared": float(result.rsquared),
        "adj_r_squared": float(result.rsquared_adj),
        "pearson_r": safe_pearson(protein_array, y_array),
        "spearman_r": safe_spearman(protein_array, y_array),
        "status": "ok",
    }


def add_fdr(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["slope_qvalue_bh"] = np.nan
    finite_mask = results["slope_pvalue"].notna() & np.isfinite(results["slope_pvalue"])
    if finite_mask.any():
        _, qvalues, _, _ = multipletests(
            results.loc[finite_mask, "slope_pvalue"].to_numpy(dtype=float),
            alpha=0.05,
            method="fdr_bh",
        )
        results.loc[finite_mask, "slope_qvalue_bh"] = qvalues
    return results


def build_summary(metadata: pd.DataFrame, results: pd.DataFrame) -> dict:
    finite_mask = results["slope_pvalue"].notna() & np.isfinite(results["slope_pvalue"])
    qvalue_mask = results["slope_qvalue_bh"].notna() & np.isfinite(results["slope_qvalue_bh"])
    status_counts = results["status"].value_counts().sort_index().to_dict()
    group_counts = metadata["group"].astype(str).value_counts().sort_index().to_dict()
    state_counts = metadata["state"].fillna("Unknown").astype(str).value_counts().sort_index().to_dict()

    return {
        "metadata_path": str(metadata.attrs.get("source_path", DEFAULT_METADATA_PATH)),
        "matrix_path": str(results.attrs.get("matrix_path", DEFAULT_MATRIX_PATH)),
        "analysis": "one_protein_one_linear_model",
        "model_formula": "fep_delta ~ protein_value",
        "target": "fep_delta",
        "sample_count": int(len(metadata)),
        "subject_count": int(metadata["sn"].astype(str).nunique()),
        "protein_count": int(len(results)),
        "finite_pvalue_count": int(finite_mask.sum()),
        "finite_qvalue_count": int(qvalue_mask.sum()),
        "p_lt_0_05_count": int((results["slope_pvalue"] < 0.05).fillna(False).sum()),
        "p_lt_0_01_count": int((results["slope_pvalue"] < 0.01).fillna(False).sum()),
        "q_lt_0_10_count": int((results["slope_qvalue_bh"] < 0.10).fillna(False).sum()),
        "q_lt_0_05_count": int((results["slope_qvalue_bh"] < 0.05).fillna(False).sum()),
        "group_counts": group_counts,
        "state_counts": state_counts,
        "timepoints_present": sorted(metadata["timepoint"].dropna().astype(float).unique().tolist()),
        "status_counts": status_counts,
        "usable_for_exploratory_feature_selection": True,
        "feature_selection_note": (
            "These per-protein p-values match the meeting-note request better than the earlier "
            "subject-by-subject slope p-values. They are suitable as exploratory ranking statistics, "
            "but repeated measures within subject mean the independence assumption is imperfect."
        ),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_path)
    metadata.attrs["source_path"] = str(args.metadata_path)
    matrix = load_matrix(args.matrix_path)
    metadata, x, y = build_dataset(metadata, matrix)

    rows = [fit_one_protein(protein_id, x[protein_id], y) for protein_id in x.columns]
    results = pd.DataFrame(rows)
    results = add_fdr(results)
    results.attrs["matrix_path"] = str(args.matrix_path)
    results["abs_slope_coef"] = results["slope_coef"].abs()
    results = results.sort_values(
        ["slope_pvalue", "slope_qvalue_bh", "abs_slope_coef", "protein_id"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    significant_hits = results.loc[
        (results["slope_pvalue"] < 0.05).fillna(False) | (results["slope_qvalue_bh"] < 0.10).fillna(False)
    ].copy()
    significant_hits.insert(0, "selection_rank", np.arange(1, len(significant_hits) + 1))

    summary = build_summary(metadata, results)

    results_path = args.output_dir / "protein_fep_delta_pvalues.csv"
    hits_path = args.output_dir / "protein_fep_delta_significant_hits.csv"
    summary_path = args.output_dir / "protein_fep_delta_pvalues_summary.json"

    results.to_csv(results_path, index=False)
    significant_hits.to_csv(hits_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved per-protein p-values to {results_path}")
    print(f"Saved significant hits to {hits_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Finite p-values: {summary['finite_pvalue_count']}")
    print(f"Proteins with p < 0.05: {summary['p_lt_0_05_count']}")
    print(f"Proteins with q < 0.10: {summary['q_lt_0_10_count']}")


if __name__ == "__main__":
    main()
