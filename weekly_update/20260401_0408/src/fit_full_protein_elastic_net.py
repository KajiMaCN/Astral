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
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


DEFAULT_METADATA_PATH = Path("/workspace/Data/old/metadata-fep_delta.csv")
DEFAULT_MATRIX_PATH = Path("/workspace/Data/20260401_0408/cmc-combat_0409.csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/weekly_update/20260401_0408/temp")
STATE_PALETTE = {"UHR": "#1f77b4", "FEP": "#d62728", "Unknown": "#7f7f7f", "Control": "#2ca02c"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the full-protein ElasticNet model and save model-specific outputs."
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    sample_col = metadata.columns[0]
    metadata = metadata.rename(columns={sample_col: "sample_id"})
    metadata["fep_delta"] = pd.to_numeric(metadata["fep_delta"], errors="coerce")
    metadata = metadata.loc[metadata["fep_delta"].notna()].copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
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
    x = matrix.loc[:, sample_ids].T

    if x.isna().any().any():
        missing_counts = x.isna().sum(axis=0)
        bad_features = missing_counts.loc[missing_counts > 0].index.tolist()
        raise ValueError(
            "Protein matrix contains missing values for the modeling subset. "
            f"Examples: {bad_features[:10]}"
        )

    y = metadata["fep_delta"].astype(float).to_numpy()
    zero_variance_mask = x.var(axis=0, ddof=0) == 0
    return metadata, x, y, zero_variance_mask


def summarize_metrics(observed: np.ndarray, predicted: np.ndarray):
    residual = observed - predicted
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    mae = float(np.mean(np.abs(residual)))
    r2 = float(r2_score(observed, predicted))
    observed_centered = observed - observed.mean()
    predicted_centered = predicted - predicted.mean()
    denom = np.sqrt(np.sum(observed_centered ** 2) * np.sum(predicted_centered ** 2))
    pearson_r = float(np.sum(observed_centered * predicted_centered) / denom) if denom > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pearson_r}


def save_annotated_plot(predictions: pd.DataFrame, selected: pd.DataFrame, summary: dict, output_path: Path):
    metrics = summarize_metrics(
        predictions["observed_fep_delta"].to_numpy(dtype=float),
        predictions["predicted_fep_delta"].to_numpy(dtype=float),
    )

    fig = plt.figure(figsize=(12.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.6, 1.9], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    for state, state_df in predictions.groupby("state", dropna=False):
        label = "Unknown" if pd.isna(state) else str(state)
        ax.scatter(
            state_df["observed_fep_delta"],
            state_df["predicted_fep_delta"],
            label=label,
            s=65,
            alpha=0.9,
            color=STATE_PALETTE.get(label, "#7f7f7f"),
            edgecolors="white",
            linewidths=0.8,
        )

    axis_min = float(min(predictions["observed_fep_delta"].min(), predictions["predicted_fep_delta"].min()))
    axis_max = float(max(predictions["observed_fep_delta"].max(), predictions["predicted_fep_delta"].max()))
    padding = (axis_max - axis_min) * 0.05 or 1.0
    lower = axis_min - padding
    upper = axis_max + padding
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2, color="black")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Observed fep_delta (months)")
    ax.set_ylabel("Predicted fep_delta (months)")
    ax.set_title(
        "elastic_net_full_fit\n"
        f"RMSE={metrics['rmse']:.2f}, "
        f"MAE={metrics['mae']:.2f}, "
        f"R^2={metrics['r2']:.2f}, "
        f"Pearson r={metrics['pearson_r']:.2f}"
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.2)

    top_df = selected.head(18)
    lines = [
        "Elastic net selected proteins",
        f"non-zero count: {summary['nonzero_coefficient_count']}",
        f"alpha: {summary['alpha']}",
        f"l1_ratio: {summary['l1_ratio']}",
        "",
    ]
    for _, row in top_df.iterrows():
        lines.append(f"{row['protein_id']}  {row['elastic_net_coef']:+.3f}")
    if len(selected) > len(top_df):
        lines.extend(["", f"... +{len(selected) - len(top_df)} more"])

    ax_text.axis("off")
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12, wspace=0.1)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_path)
    matrix = load_matrix(args.matrix_path)
    metadata, x, y, zero_variance_mask = build_dataset(metadata, matrix)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = ElasticNet(
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        max_iter=20000,
        tol=1e-3,
        selection="random",
        random_state=0,
    )
    model.fit(x_scaled, y)
    predicted = model.predict(x_scaled)

    coefficients = pd.DataFrame(
        {
            "protein_id": x.columns.to_numpy(),
            "elastic_net_coef": model.coef_,
        }
    )
    coefficients["elastic_net_abs_coef"] = coefficients["elastic_net_coef"].abs()
    coefficients["elastic_net_nonzero"] = ~np.isclose(
        coefficients["elastic_net_coef"].to_numpy(),
        0.0,
        atol=1e-12,
    )
    coefficients = coefficients.sort_values(
        ["elastic_net_nonzero", "elastic_net_abs_coef", "protein_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    selected = coefficients.loc[coefficients["elastic_net_nonzero"]].copy()
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))

    predictions = pd.DataFrame(
        {
            "sample_id": metadata["sample_id"].to_numpy(),
            "sn": metadata["sn"].astype(str).to_numpy(),
            "state": metadata["state"].fillna("Unknown").astype(str).to_numpy(),
            "timepoint": metadata["timepoint"].to_numpy(),
            "observed_fep_delta": y,
            "predicted_fep_delta": predicted,
        }
    )
    predictions["residual"] = predictions["observed_fep_delta"] - predictions["predicted_fep_delta"]

    summary = {
        "metadata_path": str(args.metadata_path),
        "matrix_path": str(args.matrix_path),
        "sample_count": int(x.shape[0]),
        "subject_count": int(metadata["sn"].nunique()),
        "protein_feature_count": int(x.shape[1]),
        "zero_variance_protein_count": int(zero_variance_mask.sum()),
        "target": "fep_delta",
        "alpha": float(args.alpha),
        "l1_ratio": float(args.l1_ratio),
        "nonzero_coefficient_count": int(selected.shape[0]),
        "train_rmse": float(np.sqrt(mean_squared_error(y, predicted))),
        "train_r2": float(r2_score(y, predicted)),
        "top_nonzero_proteins": selected["protein_id"].head(20).tolist(),
    }

    summary_path = args.output_dir / "elastic_net_full_fit_summary.json"
    coefficients_path = args.output_dir / "elastic_net_full_fit_coefficients.csv"
    predictions_path = args.output_dir / "elastic_net_full_fit_predictions.csv"
    selected_path = args.output_dir / "elastic_net_full_fit_selected_proteins.csv"
    plot_path = args.output_dir / "elastic_net_full_fit_annotated.png"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    coefficients.to_csv(coefficients_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    selected.to_csv(selected_path, index=False)
    save_annotated_plot(predictions=predictions, selected=selected, summary=summary, output_path=plot_path)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved coefficient table to {coefficients_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved selected proteins to {selected_path}")
    print(f"Saved plots to {plot_path} and {plot_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
