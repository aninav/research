"""
model.py
========
Predictive modeling for Research Question 2:
"Can textual features of macroeconomic headlines help predict the
magnitude of short-term market reactions?"

Target variable : 30-minute post-event realized volatility (SPY + QQQ)
Features        : sentiment score, keyword flags, category dummies, time features

Models:
- Linear Regression (baseline)
- Random Forest
- Gradient Boosting

Outputs (saved to ../data/tables/):
- table5_model_performance.csv   : RMSE, R², MAE for each model × ticker
- table6_feature_importance.csv  : feature importances from Random Forest
- figures/figure4_predictions.png: actual vs predicted scatter plots
- figures/figure5_feature_importance.png

Usage
-----
    python model.py

Dependencies
------------
    pip install pandas numpy scikit-learn matplotlib
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)




THIS_DIR    = Path(__file__).resolve().parent
DATA_DIR    = THIS_DIR.parent / "data"
TABLES_DIR  = DATA_DIR / "tables"
FIGURES_DIR = DATA_DIR / "figures"
MASTER_PATH = DATA_DIR / "master_dataset.csv"

TICKERS = ["SPY", "QQQ"]





def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Select all usable feature columns (exclude IDs, raw market metrics,
    and any columns with too many NaNs).
    """
    exclude_prefixes = ["SPY_", "QQQ_", "event_id", "timestamp",
                        "event_name", "event_category", "event_flag"]
    feature_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and df[c].dtype in [np.float64, np.int64, float, int]
        and df[c].notna().mean() > 0.8
    ]
    return feature_cols


def prepare_Xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return feature matrix X and target vector y, dropping rows with NaN
    in either the target or any feature.
    """
    feature_cols = get_feature_cols(df)
    keep_cols    = feature_cols + [target_col]
    clean        = df[keep_cols].dropna()

    X = clean[feature_cols]
    y = clean[target_col]
    return X, y






def build_models() -> dict:
    return {
        "Linear Regression (Ridge)": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }






def evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    """
    Cross-validated evaluation. Returns RMSE, MAE, R² (mean ± std across folds).
    Falls back to train-set metrics if dataset is too small for CV.
    """
    if len(X) < cv * 2:
        log.warning("Dataset too small for %d-fold CV (%d rows). Using train metrics.", cv, len(X))
        model.fit(X, y)
        y_pred = model.predict(X)
        return {
            "rmse"     : np.sqrt(mean_squared_error(y, y_pred)),
            "rmse_std" : np.nan,
            "mae"      : mean_absolute_error(y, y_pred),
            "r2"       : r2_score(y, y_pred),
            "r2_std"   : np.nan,
            "note"     : "train-set (too few rows for CV)",
        }

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    neg_mse = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
    neg_mae = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    r2      = cross_val_score(model, X, y, cv=kf, scoring="r2")


    model.fit(X, y)

    return {
        "rmse"     : np.sqrt(-neg_mse.mean()),
        "rmse_std" : np.sqrt(neg_mse.std()),
        "mae"      : -neg_mae.mean(),
        "r2"       : r2.mean(),
        "r2_std"   : r2.std(),
        "note"     : f"{cv}-fold CV",
    }






def table5_model_performance(df: pd.DataFrame) -> pd.DataFrame:
    rows   = []
    models = build_models()

    for ticker in TICKERS:
        target = f"{ticker}_rv_post_30m"
        X, y   = prepare_Xy(df, target)

        if len(X) < 10:
            log.warning("Too few samples for %s (%d rows). Skipping.", ticker, len(X))
            continue

        log.info("Training models for target: %s (n=%d, features=%d)", target, len(X), X.shape[1])

        for model_name, model in models.items():
            metrics = evaluate_model(model, X.copy(), y.copy())
            rows.append({
                "ticker"    : ticker,
                "target"    : target,
                "model"     : model_name,
                "n_samples" : len(X),
                "n_features": X.shape[1],
                **metrics,
            })
            log.info("  %s | RMSE: %.6f | R²: %.4f", model_name, metrics["rmse"], metrics["r2"])

    result = pd.DataFrame(rows).round(6)
    out = TABLES_DIR / "table5_model_performance.csv"
    result.to_csv(out, index=False)
    log.info("Table 5 saved: %s", out)
    return result






def table6_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in TICKERS:
        target = f"{ticker}_rv_post_30m"
        X, y   = prepare_Xy(df, target)

        if len(X) < 10:
            continue

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=5,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        rf.fit(X, y)

        importance_df = pd.DataFrame({
            "ticker"    : ticker,
            "feature"   : X.columns,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)

        rows.append(importance_df)

    if not rows:
        log.warning("No feature importance data generated.")
        return pd.DataFrame()

    result = pd.concat(rows).round(6)
    out = TABLES_DIR / "table6_feature_importance.csv"
    result.to_csv(out, index=False)
    log.info("Table 6 saved: %s", out)
    return result






def figure4_predictions(df: pd.DataFrame) -> None:
    """
    Actual vs predicted scatter for best model (Random Forest) per ticker.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Actual vs Predicted: 30-min Post-Event Realized Volatility", fontsize=12)

    for ax, ticker in zip(axes, TICKERS):
        target = f"{ticker}_rv_post_30m"
        X, y   = prepare_Xy(df, target)

        if len(X) < 5:
            ax.set_title(f"{ticker} (insufficient data)")
            continue

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=5,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        rf.fit(X, y)
        y_pred = rf.predict(X)

        ax.scatter(y, y_pred, alpha=0.6, color="#4C8BE8", edgecolors="white", s=40)
        lim = [min(y.min(), y_pred.min()) * 0.95, max(y.max(), y_pred.max()) * 1.05]
        ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5, label="Perfect fit")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual RV")
        ax.set_ylabel("Predicted RV")
        ax.set_title(f"{ticker}  |  R² = {r2_score(y, y_pred):.3f}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "figure4_predictions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure 4 saved: %s", out)


def figure5_feature_importance(df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of top-10 features by importance (Random Forest).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Top Feature Importances — Random Forest (target: 30-min RV)", fontsize=12)

    for ax, ticker in zip(axes, TICKERS):
        target = f"{ticker}_rv_post_30m"
        X, y   = prepare_Xy(df, target)

        if len(X) < 5:
            ax.set_title(f"{ticker} (insufficient data)")
            continue

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=5,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        rf.fit(X, y)

        importance_df = (
            pd.Series(rf.feature_importances_, index=X.columns)
            .sort_values(ascending=True)
            .tail(10)
        )

        ax.barh(importance_df.index, importance_df.values,
                color="#4C8BE8", alpha=0.85, edgecolor="white")
        ax.set_title(ticker, fontsize=12)
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "figure5_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure 5 saved: %s", out)






def run_models() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MASTER_PATH)
    log.info("Loaded master dataset: %d rows × %d cols", len(df), len(df.columns))

    log.info("=== Training models ===")
    t5 = table5_model_performance(df)
    t6 = table6_feature_importance(df)

    log.info("=== Generating figures ===")
    figure4_predictions(df)
    figure5_feature_importance(df)

    log.info("=== Modeling complete ===")
    if not t5.empty:
        print("\n=== Model Performance Summary ===")
        print(t5[["ticker", "model", "n_samples", "rmse", "r2", "note"]].to_string(index=False))

    print(f"\nAll tables saved to: {TABLES_DIR}")
    print(f"All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_models()
