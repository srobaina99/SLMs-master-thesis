"""
XGBoost hyperparameter tuning for full_xgb experiment.
Uses 5-fold CV on train, then evaluates best params on dev.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr
from itertools import product
import xgboost as xgb

# Reuse feature building from the main experiment file
from feature_experiments import build_features, L1S, DATA_DIR

FULL_FEATURES = [
    "word_len", "norm_edit_dist", "pos", "suffix", "prefix",
    "shared_prefix_ratio", "context_len", "max_consonant_cluster", "double_letter",
    "jaro_winkler", "ngram_overlap", "first_letter_match", "zipf_freq",
]

# ── Parameter grid ──
PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.85],
    "colsample_bytree": [0.7, 0.85],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 2.0],
}

N_FOLDS = 5


def cv_score(params, X, y):
    """Return mean RMSE across K folds."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        model = xgb.XGBRegressor(random_state=42, verbosity=0, **params)
        model.fit(X.iloc[train_idx], y[train_idx],
                  eval_set=[(X.iloc[val_idx], y[val_idx])],
                  verbose=False)
        preds = model.predict(X.iloc[val_idx])
        rmses.append(root_mean_squared_error(y[val_idx], preds))
    return np.mean(rmses)


def random_search(X, y, l1, n_iter=80):
    """Random search over parameter grid."""
    rng = np.random.RandomState(42)
    keys = list(PARAM_GRID.keys())
    best_score = float("inf")
    best_params = None
    results = []

    for i in range(n_iter):
        params = {k: rng.choice(PARAM_GRID[k]) for k in keys}
        # Convert numpy types to Python native for xgboost
        params = {k: v.item() if hasattr(v, 'item') else v for k, v in params.items()}
        score = cv_score(params, X, y)
        results.append({**params, "cv_rmse": score})

        if score < best_score:
            best_score = score
            best_params = params.copy()

        if (i + 1) % 10 == 0:
            print(f"  [{l1}] {i+1}/{n_iter} done — best CV RMSE: {best_score:.4f}")

    return best_params, best_score, pd.DataFrame(results)


def main():
    print("Loading data...")
    data = {}
    for l1 in L1S:
        data[l1] = {
            "train": pd.read_csv(DATA_DIR / "train" / l1 / f"kvl_shared_task_{l1}_train.csv"),
            "dev": pd.read_csv(DATA_DIR / "dev" / l1 / f"kvl_shared_task_{l1}_dev.csv"),
        }

    print(f"\nParameter grid: {np.prod([len(v) for v in PARAM_GRID.values()])} total combos")
    print(f"Random search: 80 iterations x {N_FOLDS}-fold CV per L1\n")

    for l1 in L1S:
        print(f"{'='*60}")
        print(f"Tuning {l1.upper()}")
        print(f"{'='*60}")

        train_df = data[l1]["train"]
        dev_df = data[l1]["dev"]

        X_train = build_features(train_df, FULL_FEATURES)
        y_train = train_df["GLMM_score"].values
        X_dev = build_features(dev_df, FULL_FEATURES)
        y_dev = dev_df["GLMM_score"].values

        best_params, best_cv, results_df = random_search(X_train, y_train, l1, n_iter=80)

        print(f"\n  Best CV RMSE: {best_cv:.4f}")
        print(f"  Best params: {best_params}")

        # Train on full train set with best params, evaluate on dev
        model = xgb.XGBRegressor(random_state=42, verbosity=0, **best_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)
        dev_rmse = root_mean_squared_error(y_dev, preds)
        dev_r, _ = pearsonr(preds, y_dev)
        print(f"  Dev RMSE: {dev_rmse:.4f}  |  Dev Pearson: {dev_r:.4f}")

        # Current baseline comparison
        print(f"  (Previous full_xgb: {'1.464' if l1 == 'es' else '1.421'})")

        # Top 5 param combos
        top5 = results_df.nsmallest(5, "cv_rmse")
        print(f"\n  Top 5 configurations:")
        for _, row in top5.iterrows():
            print(f"    CV RMSE={row['cv_rmse']:.4f} | "
                  f"n_est={int(row['n_estimators'])} depth={int(row['max_depth'])} "
                  f"lr={row['learning_rate']:.3f} sub={row['subsample']:.2f} "
                  f"col={row['colsample_bytree']:.2f} a={row['reg_alpha']:.1f} l={row['reg_lambda']:.1f}")

        # Feature importance
        print(f"\n  Feature importance (gain):")
        imp = model.feature_importances_
        feat_names = X_train.columns
        for idx in np.argsort(imp)[::-1][:10]:
            print(f"    {feat_names[idx]:<25} {imp[idx]:.4f}")

        results_df.to_csv(f"results/tune_xgb_{l1}.csv", index=False)
        print()


if __name__ == "__main__":
    main()
