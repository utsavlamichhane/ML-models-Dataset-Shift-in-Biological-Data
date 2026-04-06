#!/usr/bin/env python3

import os
import sys
import time
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    LeaveOneGroupOut,
    cross_val_predict,
    learning_curve,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
from sklearn.impute import SimpleImputer
import joblib
import shap

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(".")

DATASETS = {
    "Level5": "preprocessed_data_Level5.xlsx",
    "Level6": "preprocessed_data_Level6.xlsx",
    "Level7": "preprocessed_data_Level7.xlsx",
}

PARAM_GRID = {
    "n_estimators":      [500, 1000, 1500, 2000],
    "max_depth":         [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 0.3, 0.5, None],
}

N_GRID_COMBOS = 1
for v in PARAM_GRID.values():
    N_GRID_COMBOS *= len(v)

TOP_FEATURES_PLOT = 30
N_EXTREME = 10
LEARNING_CURVE_SIZES = np.linspace(0.1, 1.0, 10)


def load_dataset(filepath):
    log.info(f"Loading {filepath}")
    df = pd.read_excel(filepath)

    sample_ids = df["SampleID"].values
    groups = df["Farm_Code"].values
    y = df["ADG"].values

    X = df.drop(columns=["SampleID", "Farm_Code", "ADG"])

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.Categorical(X[col]).codes

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    log.info(
        f"  X shape={X.shape}  y range=[{y.min():.2f}, {y.max():.2f}]  "
        f"farms={sorted(np.unique(groups))}"
    )
    return X, y, groups, sample_ids


def _compute_metrics(y_true, y_pred):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else float("nan")
    safe_y = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / safe_y)) * 100
    pr, pp = stats.pearsonr(y_true, y_pred)
    sr, sp = stats.spearmanr(y_true, y_pred)

    return {
        "N": n,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Adjusted_R2": adj_r2,
        "Explained_Variance": evs,
        "MAPE_pct": mape,
        "Pearson_r": pr,
        "Pearson_p": pp,
        "Spearman_r": sr,
        "Spearman_p": sp,
    }


def _format_metrics(d):
    lines = []
    for k, v in d.items():
        if isinstance(v, float):
            lines.append(f"{k:25s}: {v:.6f}")
        else:
            lines.append(f"{k:25s}: {v}")
    return "\n".join(lines)


def _style():
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    log.info(f"    saved {path.name}")


def plot_actual_vs_predicted(y, yhat, out, groups=None):
    _style()
    fig, ax = plt.subplots(figsize=(8, 8))

    if groups is not None:
        ugroups = sorted(np.unique(groups))
        palette = sns.color_palette("tab10", len(ugroups))
        for g, c in zip(ugroups, palette):
            m = groups == g
            ax.scatter(
                y[m], yhat[m], s=30, alpha=0.6, color=c,
                label=f"Farm {g}", edgecolors="w", linewidths=0.3,
            )
        ax.legend(title="Farm", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.scatter(y, yhat, s=30, alpha=0.5, c="steelblue", edgecolors="w", linewidths=0.3)

    lo = min(y.min(), yhat.min()) - 0.3
    hi = max(y.max(), yhat.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--r", lw=1.5, label="Identity")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual ADG")
    ax.set_ylabel("Predicted ADG")
    r2 = r2_score(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    ax.set_title(f"Actual vs Predicted  (R²={r2:.3f}, RMSE={rmse:.3f})")
    _save(fig, out / "actual_vs_predicted.png")


def plot_residuals_hist(y, yhat, out):
    _style()
    res = y - yhat
    fig, ax = plt.subplots(figsize=(8, 6))
    n_bins = min(40, max(10, len(res) // 5))
    ax.hist(res, bins=n_bins, density=True, alpha=0.7, color="steelblue", edgecolor="w")
    mu, sig = res.mean(), res.std()
    xs = np.linspace(res.min(), res.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, mu, sig), "r-", lw=2, label=f"N({mu:.3f}, {sig:.3f}²)")
    ax.axvline(0, color="k", ls="--", lw=1)
    sample = res if len(res) <= 5000 else np.random.choice(res, 5000, replace=False)
    sw, sp = stats.shapiro(sample)
    ax.text(
        0.02, 0.95,
        f"Shapiro-Wilk: W={sw:.4f}, p={sp:.2e}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.5),
    )
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Density")
    ax.set_title("Residuals Distribution")
    ax.legend()
    _save(fig, out / "residuals_hist.png")


def plot_residuals_vs_pred(y, yhat, out):
    _style()
    res = y - yhat
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(yhat, res, s=25, alpha=0.5, c="steelblue", edgecolors="w", linewidths=0.3)
    ax.axhline(0, color="r", ls="--", lw=1.5)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(res, yhat, frac=0.3)
        ax.plot(sm[:, 0], sm[:, 1], c="orange", lw=2, label="LOWESS")
        ax.legend()
    except ImportError:
        pass
    ax.set_xlabel("Predicted ADG")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residuals vs Predicted")
    _save(fig, out / "residuals_vs_pred.png")


def plot_train_test_scatter(y, y_train_pred, y_test_pred, out):
    _style()
    fig, ax = plt.subplots(figsize=(8, 8))
    train_r2 = r2_score(y, y_train_pred)
    test_r2 = r2_score(y, y_test_pred)
    ax.scatter(
        y, y_train_pred, s=25, alpha=0.35, c="dodgerblue",
        edgecolors="w", linewidths=0.2,
        label=f"Train — all-data fit (R²={train_r2:.3f})",
    )
    ax.scatter(
        y, y_test_pred, s=25, alpha=0.55, c="orangered",
        edgecolors="w", linewidths=0.2,
        label=f"Test — CV out-of-fold (R²={test_r2:.3f})",
    )
    lo = min(y.min(), y_train_pred.min(), y_test_pred.min()) - 0.3
    hi = max(y.max(), y_train_pred.max(), y_test_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--k", lw=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual ADG")
    ax.set_ylabel("Predicted ADG")
    ax.set_title("Train vs Test Predictions")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, out / "train_test_scatter.png")


def plot_feature_importances(model, feature_names, out, top_n=TOP_FEATURES_PLOT):
    _style()
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    fi_df.to_csv(out / "feature_importances.csv", index=False)
    log.info("    saved feature_importances.csv")
    top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.30)))
    ax.barh(range(len(top)), top["Importance"].values, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["Feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity (MDI)")
    ax.set_title(f"Top {top_n} Feature Importances")
    _save(fig, out / "feature_importances.png")


def plot_learning_curve_fig(X, y, best_params, cv, groups, out, n_jobs):
    _style()
    est = RandomForestRegressor(**best_params, random_state=SEED, n_jobs=1)
    sizes, tr_scores, te_scores = learning_curve(
        est, X, y,
        cv=cv,
        groups=groups,
        train_sizes=LEARNING_CURVE_SIZES,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        verbose=0,
    )
    tr_rmse = np.sqrt(-tr_scores)
    te_rmse = np.sqrt(-te_scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(sizes, tr_rmse.mean(1) - tr_rmse.std(1), tr_rmse.mean(1) + tr_rmse.std(1), alpha=0.12, color="blue")
    ax.fill_between(sizes, te_rmse.mean(1) - te_rmse.std(1), te_rmse.mean(1) + te_rmse.std(1), alpha=0.12, color="orange")
    ax.plot(sizes, tr_rmse.mean(1), "o-", c="blue", label="Training RMSE")
    ax.plot(sizes, te_rmse.mean(1), "o-", c="orange", label="Validation RMSE")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("RMSE")
    ax.set_title("Learning Curve")
    ax.legend(loc="best")
    _save(fig, out / "learning_curve.png")


def run_shap_analysis(model, X, y, yhat, sample_ids, out):
    log.info("  SHAP TreeExplainer …")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    shap.summary_plot(sv, X, show=False, max_display=30)
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    fig.savefig(out / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    log.info("    saved shap_summary.png")

    res = y - yhat
    n_ext = min(N_EXTREME, len(y) // 2)
    under_idx = np.argsort(res)[-n_ext:][::-1]
    over_idx = np.argsort(res)[:n_ext]
    idx = np.concatenate([under_idx, over_idx])

    df = pd.DataFrame(sv[idx], columns=X.columns)
    df.insert(0, "SampleID", sample_ids[idx])
    df.insert(1, "Actual_ADG", y[idx])
    df.insert(2, "Predicted_ADG", yhat[idx])
    df.insert(3, "Residual", res[idx])
    df.insert(4, "Category", ["Under-predicted"] * n_ext + ["Over-predicted"] * n_ext)
    df.to_csv(out / "extreme_sample_shap.csv", index=False)
    log.info("    saved extreme_sample_shap.csv")
    return sv


def generate_per_farm_results(y, y_test_pred, y_train_pred, groups, sample_ids, shap_values, X, out_dir):
    per_farm_dir = Path(out_dir) / "per_farm"
    per_farm_dir.mkdir(parents=True, exist_ok=True)
    log.info("  Generating per-farm breakdown …")

    unique_farms = sorted(np.unique(groups))
    all_farm_metrics = []

    for farm in unique_farms:
        farm_dir = per_farm_dir / f"Farm_{farm}"
        farm_dir.mkdir(parents=True, exist_ok=True)

        mask = groups == farm
        n_f = int(mask.sum())
        y_f = y[mask]
        yhat_f = y_test_pred[mask]
        ytrain_f = y_train_pred[mask]

        log.info(f"    Farm {farm}  (n={n_f})")

        fm = _compute_metrics(y_f, yhat_f)
        comp_entry = {"Farm": farm, "N_samples": n_f}
        comp_entry.update(fm)
        all_farm_metrics.append(comp_entry)

        with open(farm_dir / "metrics.txt", "w") as f:
            f.write(f"Per-Farm Test Metrics — Farm {farm}\n")
            f.write("(This farm was the held-out test set; model trained on all other farms)\n")
            f.write("=" * 55 + "\n\n")
            f.write(_format_metrics(fm) + "\n")

        plot_actual_vs_predicted(y_f, yhat_f, farm_dir)
        plot_residuals_vs_pred(y_f, yhat_f, farm_dir)
        if n_f >= 10:
            plot_residuals_hist(y_f, yhat_f, farm_dir)
        plot_train_test_scatter(y_f, ytrain_f, yhat_f, farm_dir)

        if shap_values is not None:
            sv_f = shap_values[mask]
            X_f = X.iloc[np.where(mask)[0]]
            try:
                shap.summary_plot(sv_f, X_f, show=False, max_display=20)
                fig = plt.gcf()
                fig.set_size_inches(12, 8)
                fig.suptitle(f"SHAP Summary — Farm {farm}  (n={n_f})", y=1.02)
                fig.savefig(farm_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
                plt.close("all")
            except Exception as exc:
                log.warning(f"      SHAP plot failed for Farm {farm}: {exc}")
                plt.close("all")

    comp_df = pd.DataFrame(all_farm_metrics)
    col_order = [
        "Farm", "N_samples", "N", "R2", "Adjusted_R2", "RMSE", "MAE",
        "Explained_Variance", "MAPE_pct", "Pearson_r", "Pearson_p",
        "Spearman_r", "Spearman_p",
    ]
    comp_df = comp_df[[c for c in col_order if c in comp_df.columns]]
    comp_df.to_csv(per_farm_dir / "farm_comparison_metrics.csv", index=False)
    log.info("    saved farm_comparison_metrics.csv")

    _plot_farm_comparison(comp_df, per_farm_dir)
    log.info("  Per-farm breakdown complete.")


def _plot_farm_comparison(comp_df, out):
    _style()
    metrics_to_plot = [
        ("R2", "R²"),
        ("RMSE", "RMSE"),
        ("MAE", "MAE"),
        ("Pearson_r", "Pearson r"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = sns.color_palette("tab10", len(comp_df))

    for ax, (col, label) in zip(axes.flat, metrics_to_plot):
        bars = ax.bar(
            comp_df["Farm"].astype(str), comp_df[col],
            color=palette, edgecolor="white", linewidth=0.5,
        )
        for bar, (_, row) in zip(bars, comp_df.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={int(row['N_samples'])}",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_ylabel(label)
        ax.set_xlabel("Farm")
        ax.set_title(f"{label} by Farm")

    fig.suptitle("Performance Comparison Across Farms  (Leave-One-Farm-Out)", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, out / "farm_comparison_barplot.png")


def run_pipeline(X, y, groups, sample_ids, cv_type, out_dir, n_jobs=-1):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feature_names = list(X.columns)

    log.info(f"  [{cv_type}] Grid search over {N_GRID_COMBOS} param combos …")
    t0 = time.time()
    rf_base = RandomForestRegressor(random_state=SEED, n_jobs=1)

    if cv_type == "random":
        gs_cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=SEED)
        gs_groups = None
    else:
        gs_cv = LeaveOneGroupOut()
        gs_groups = groups

    gs = GridSearchCV(
        estimator=rf_base,
        param_grid=PARAM_GRID,
        cv=gs_cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
        refit=True,
    )
    gs.fit(X, y, groups=gs_groups)
    elapsed_gs = time.time() - t0
    best_params = gs.best_params_
    best_cv_rmse = np.sqrt(-gs.best_score_)

    log.info(f"  Grid search done in {elapsed_gs / 60:.1f} min  |  best CV RMSE = {best_cv_rmse:.4f}")
    log.info(f"  Best params: {best_params}")

    cv_res_df = pd.DataFrame(gs.cv_results_)
    cv_res_df.to_csv(out / "cv_results.csv", index=False)

    with open(out / "best_params.txt", "w") as f:
        f.write(f"Grid-Search Best Parameters  ({cv_type})\n")
        f.write("=" * 55 + "\n\n")
        for k, v in sorted(best_params.items()):
            f.write(f"  {k:30s}: {v}\n")
        f.write(f"\nBest CV RMSE : {best_cv_rmse:.6f}\n")
        f.write(f"Grid combos  : {N_GRID_COMBOS}\n")
        cv_label = "RepeatedKFold(10, 3)" if cv_type == "random" else "LeaveOneGroupOut (10 farms)"
        f.write(f"CV strategy  : {cv_label}\n")
        f.write(f"Search time  : {elapsed_gs / 60:.1f} min\n")

    log.info("  Generating out-of-fold predictions …")
    if cv_type == "random":
        eval_cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
        eval_groups = None
    else:
        eval_cv = LeaveOneGroupOut()
        eval_groups = groups

    best_rf_cv = RandomForestRegressor(**best_params, random_state=SEED, n_jobs=1)
    y_test_pred = cross_val_predict(best_rf_cv, X, y, cv=eval_cv, groups=eval_groups, n_jobs=n_jobs)

    log.info("  Training final model on all data …")
    final_rf = RandomForestRegressor(**best_params, random_state=SEED, oob_score=True, n_jobs=n_jobs)
    final_rf.fit(X, y)
    y_train_pred = final_rf.predict(X)
    joblib.dump(final_rf, out / "best_model.joblib")

    test_metrics = _compute_metrics(y, y_test_pred)
    cv_label = "10-fold KFold" if cv_type == "random" else "Leave-One-Farm-Out"
    with open(out / "metrics.txt", "w") as f:
        f.write(f"Cross-Validated Test Metrics  ({cv_label})\n")
        f.write("=" * 55 + "\n\n")
        f.write(_format_metrics(test_metrics) + "\n")

    if hasattr(final_rf, "oob_prediction_"):
        oob_metrics = _compute_metrics(y, final_rf.oob_prediction_)
        with open(out / "oob_metrics.txt", "w") as f:
            f.write("Out-of-Bag Metrics  (final model, all data)\n")
            f.write("=" * 55 + "\n\n")
            f.write(_format_metrics(oob_metrics) + "\n")

    log.info("  Generating diagnostic plots …")
    grp_for_plot = groups if cv_type == "farm_batch" else None
    plot_actual_vs_predicted(y, y_test_pred, out, groups=grp_for_plot)
    plot_residuals_hist(y, y_test_pred, out)
    plot_residuals_vs_pred(y, y_test_pred, out)
    plot_train_test_scatter(y, y_train_pred, y_test_pred, out)
    plot_feature_importances(final_rf, feature_names, out)

    log.info("  Computing learning curve …")
    if cv_type == "random":
        lc_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        lc_groups = None
    else:
        lc_cv = LeaveOneGroupOut()
        lc_groups = groups
    plot_learning_curve_fig(X, y, best_params, lc_cv, lc_groups, out, n_jobs)

    shap_values = run_shap_analysis(final_rf, X, y, y_test_pred, sample_ids, out)

    if cv_type == "farm_batch":
        generate_per_farm_results(
            y, y_test_pred, y_train_pred, groups,
            sample_ids, shap_values, X, out,
        )

    log.info(f"  Pipeline complete → {out}\n")


def main():
    log.info("=" * 60)
    log.info("  ADG Random Forest Prediction Pipeline")
    log.info("=" * 60)

    total_t0 = time.time()

    for level, fname in DATASETS.items():
        fpath = BASE_DIR / fname
        if not fpath.exists():
            log.warning(f"  File not found: {fpath} — skipping {level}")
            continue

        X, y, groups, sids = load_dataset(str(fpath))
        log.info(f"\n{'─' * 60}")
        log.info(f"  Dataset: {level}  ({X.shape[0]} samples × {X.shape[1]} features)")
        log.info(f"{'─' * 60}")

        log.info(f"\n  ▶ RandomCV / {level}")
        run_pipeline(X, y, groups, sids, cv_type="random", out_dir=BASE_DIR / "randomCV" / level)

        log.info(f"\n  ▶ FarmBatchCV / {level}")
        run_pipeline(X, y, groups, sids, cv_type="farm_batch", out_dir=BASE_DIR / "FarmBatchCV" / level)

    elapsed_total = time.time() - total_t0
    log.info("=" * 60)
    log.info(f"  All pipelines complete.  Total wall time: {elapsed_total / 3600:.1f} h")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
