"""
Ablation: start from all 9 winning features, drop one at a time.
Then test a "clean" set with only the non-redundant winners.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

from feature_experiments import build_features, DATA_DIR
from test_es_features import compute_new_features

BASE_FEATURES = [
    "word_len", "norm_edit_dist", "pos", "suffix", "prefix",
    "shared_prefix_ratio", "context_len", "max_consonant_cluster", "double_letter",
    "jaro_winkler", "ngram_overlap", "first_letter_match", "zipf_freq",
]

ALL_WINNERS = [
    "suffix_transform_match", "is_multiword_source", "source_word_len",
    "len_ratio", "len_diff", "source_has_accent", "shared_suffix_ratio",
    "is_latinate", "syllable_count",
]

XGB_PARAMS = {
    "n_estimators": 600, "max_depth": 3, "learning_rate": 0.02,
    "subsample": 0.85, "colsample_bytree": 0.7, "reg_alpha": 0.5,
    "reg_lambda": 2.0, "random_state": 42, "verbosity": 0,
}


def cv_rmse(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmses = []
    for ti, vi in kf.split(X):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[ti], y[ti], verbose=False)
        rmses.append(root_mean_squared_error(y[vi], m.predict(X.iloc[vi])))
    return np.mean(rmses)


def build_full(train_df, dev_df, new_feats):
    X_tr = pd.concat([build_features(train_df, BASE_FEATURES),
                       compute_new_features(train_df, new_feats, l1="es")], axis=1)
    X_dv = pd.concat([build_features(dev_df, BASE_FEATURES),
                       compute_new_features(dev_df, new_feats, l1="es")], axis=1)
    return X_tr, X_dv


def main():
    l1 = "es"
    train_df = pd.read_csv(DATA_DIR / "train" / l1 / f"kvl_shared_task_{l1}_train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev" / l1 / f"kvl_shared_task_{l1}_dev.csv")
    y_train = train_df["GLMM_score"].values
    y_dev = dev_df["GLMM_score"].values

    # ── Full winners baseline ──
    X_tr, X_dv = build_full(train_df, dev_df, ALL_WINNERS)
    full_cv = cv_rmse(X_tr, y_train)
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr, y_train, verbose=False)
    full_dev = root_mean_squared_error(y_dev, m.predict(X_dv))

    print(f"All 9 winners:  CV = {full_cv:.4f}  |  Dev = {full_dev:.4f}")
    print(f"\n{'Dropped feature':<28} {'CV RMSE':>8} {'Delta':>8} {'Dev RMSE':>9} {'Delta':>8}")
    print("-" * 65)

    # ── Drop-one ablation ──
    drop_results = {}
    for feat in ALL_WINNERS:
        subset = [f for f in ALL_WINNERS if f != feat]
        X_tr, X_dv = build_full(train_df, dev_df, subset)
        feat_cv = cv_rmse(X_tr, y_train)
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr, y_train, verbose=False)
        feat_dev = root_mean_squared_error(y_dev, m.predict(X_dv))
        cv_d = feat_cv - full_cv
        dev_d = feat_dev - full_dev
        marker = " DROP" if cv_d < -0.0005 else (" KEEP" if cv_d > 0.001 else "")
        print(f"- {feat:<26} {feat_cv:>8.4f} {cv_d:>+8.4f} {feat_dev:>9.4f} {dev_d:>+8.4f}{marker}")
        drop_results[feat] = cv_d

    # ── Identify features to drop (removing them improves or barely hurts CV) ──
    to_drop = [f for f, d in drop_results.items() if d < -0.0005]
    clean_set = [f for f in ALL_WINNERS if f not in to_drop]

    print(f"\n{'='*65}")
    if to_drop:
        print(f"Dropping: {', '.join(to_drop)}")
    print(f"Clean set: {', '.join(clean_set)}")

    X_tr, X_dv = build_full(train_df, dev_df, clean_set)
    clean_cv = cv_rmse(X_tr, y_train)
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr, y_train, verbose=False)
    clean_dev = root_mean_squared_error(y_dev, m.predict(X_dv))
    print(f"Clean CV = {clean_cv:.4f}  |  Clean Dev = {clean_dev:.4f}")

    # ── Also try the hand-picked "top 5" set ──
    top5 = ["len_diff", "source_word_len", "is_multiword_source", "is_latinate", "syllable_count"]
    print(f"\n{'='*65}")
    print(f"Hand-picked top 5: {', '.join(top5)}")
    X_tr, X_dv = build_full(train_df, dev_df, top5)
    t5_cv = cv_rmse(X_tr, y_train)
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr, y_train, verbose=False)
    t5_dev = root_mean_squared_error(y_dev, m.predict(X_dv))
    print(f"Top5 CV = {t5_cv:.4f}  |  Top5 Dev = {t5_dev:.4f}")

    # ── Summary ──
    print(f"\n{'='*65}")
    print("Summary:")
    print(f"  Original full_xgb (no new feats):  Dev = 1.4633")
    print(f"  All 9 winners:                     Dev = {full_dev:.4f}")
    print(f"  Clean set ({len(clean_set)} feats):              Dev = {clean_dev:.4f}")
    print(f"  Hand-picked top 5:                 Dev = {t5_dev:.4f}")


if __name__ == "__main__":
    main()
