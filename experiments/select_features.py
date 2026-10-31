"""
Feature-set selection on validation days.
Run:  python -m experiments.select_features

Only training days are used: fit on days 1-7, compare on days 8-10.
The held-out days 11-14 are not touched here.
"""

import warnings

import numpy as np

from data_generator import KPI_FEATURES, generate_dataset
from evaluation import evaluate_flags, format_table
from features import RESIDUAL_COLUMNS, SMOOTHED_COLUMNS, build_features
from model_trainer import fit_model, train_mask

warnings.filterwarnings("ignore")


def main():
    df = generate_dataset().sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    feats = build_features(df)
    X = feats.copy()

    fit_days = train_mask(df, 7)
    val_days = train_mask(df, 10) & ~fit_days

    # Residuals divided by each cell's typical residual size (MAD on the fit days)
    res_cols = RESIDUAL_COLUMNS + SMOOTHED_COLUMNS
    mad = (feats.loc[fit_days, res_cols].groupby(df.loc[fit_days, "cell_id"])
           .agg(lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))))
    scaled = feats[res_cols] / mad.loc[df["cell_id"]].clip(lower=1e-3).to_numpy()
    scaled_cols = [f"{c}_z" for c in res_cols]
    X[scaled_cols] = scaled.to_numpy()

    variants = {
        "raw": KPI_FEATURES,
        "residual": RESIDUAL_COLUMNS + SMOOTHED_COLUMNS,
        "raw+residual": KPI_FEATURES + RESIDUAL_COLUMNS,
        "raw+smoothed": KPI_FEATURES + SMOOTHED_COLUMNS,
        "residual_cellscaled": scaled_cols,
    }
    results = {}
    for name, cols in variants.items():
        model, scaler = fit_model(X.loc[fit_days, cols].to_numpy())
        flags = model.decision_function(scaler.transform(X[cols].to_numpy())) < 0
        results[name] = evaluate_flags(df, flags, val_days)

    print("VALIDATION days 8-10 (fit on days 1-7)\n" + format_table(results))


if __name__ == "__main__":
    main()
