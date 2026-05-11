"""
Test ES-targeted features individually and in combination.
Uses 5-fold CV on train to measure each feature's marginal contribution
on top of the current full_xgb feature set.
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr
import xgboost as xgb

from feature_experiments import build_features, DATA_DIR

# ── Current full_xgb features (baseline to compare against) ──
BASE_FEATURES = [
    "word_len", "norm_edit_dist", "pos", "suffix", "prefix",
    "shared_prefix_ratio", "context_len", "max_consonant_cluster", "double_letter",
    "jaro_winkler", "ngram_overlap", "first_letter_match", "zipf_freq",
]

# ── ES cognate suffix transformation rules ──
ES_SUFFIX_RULES = [
    ("tion", "ción"), ("sion", "sión"), ("ty", "dad"), ("ism", "ismo"),
    ("ist", "ista"), ("al", "al"), ("ble", "ble"), ("ous", "oso"),
    ("ive", "ivo"), ("ence", "encia"), ("ance", "ancia"), ("ly", "mente"),
    ("ment", "mento"), ("ant", "ante"), ("ent", "ente"),
]

# ── New feature extraction functions ──

def compute_new_features(df, feature_keys, l1="es"):
    """Compute new ES-targeted features. Returns DataFrame with new columns only."""
    feats = {}
    en = df["en_target_word"].str.lower()
    src = df["L1_source_word"].astype(str).str.lower()

    if "suffix_transform_match" in feature_keys:
        if l1 == "es":
            rules = ES_SUFFIX_RULES
        else:
            rules = []  # DE rules could be added later

        def check_suffix_match(row):
            e, s = row["en_target_word"].lower(), str(row["L1_source_word"]).lower()
            for en_sfx, l1_sfx in rules:
                if e.endswith(en_sfx) and s.endswith(l1_sfx):
                    return 1
            return 0
        feats["suffix_transform_match"] = df.apply(check_suffix_match, axis=1)

    if "is_multiword_source" in feature_keys:
        feats["is_multiword_source"] = src.str.contains(r'\s').astype(int)

    if "source_word_len" in feature_keys:
        feats["source_word_len"] = src.str.len()

    if "len_ratio" in feature_keys:
        feats["len_ratio"] = src.str.len() / en.str.len().clip(lower=1)

    if "len_diff" in feature_keys:
        feats["len_diff"] = src.str.len() - en.str.len()

    if "source_has_accent" in feature_keys:
        feats["source_has_accent"] = src.str.contains(r'[áéíóúàèìòùäëïöü]').astype(int)

    if "shared_suffix_ratio" in feature_keys:
        def _shared_suffix(row):
            e = row["en_target_word"].lower()[::-1]
            s = str(row["L1_source_word"]).lower()[::-1]
            shared = 0
            for a, b in zip(e, s):
                if a == b:
                    shared += 1
                else:
                    break
            return shared / max(len(e), len(s), 1)
        feats["shared_suffix_ratio"] = df.apply(_shared_suffix, axis=1)

    if "is_latinate" in feature_keys:
        latin_patterns = re.compile(r'(tion|ment|ence|ance|ible|able|ious|eous|ous|ive|ant|ent|ism|ist|ure|ary|ory)')
        feats["is_latinate"] = en.apply(lambda w: int(bool(latin_patterns.search(w))))

    if "syllable_count" in feature_keys:
        def count_syllables(word):
            word = word.lower()
            count = len(re.findall(r'[aeiouy]+', word))
            if word.endswith('e') and not word.endswith('le'):
                count = max(1, count - 1)
            return max(1, count)
        feats["syllable_count"] = en.apply(count_syllables)

    if "unique_char_ratio" in feature_keys:
        feats["unique_char_ratio"] = en.apply(lambda w: len(set(w)) / max(len(w), 1))

    return pd.DataFrame(feats)


# ── XGBoost params (from tuning, ES-optimized) ──
XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 3,
    "learning_rate": 0.02,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "verbosity": 0,
}


def cv_rmse(X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X.iloc[train_idx], y[train_idx], verbose=False)
        preds = model.predict(X.iloc[val_idx])
        rmses.append(root_mean_squared_error(y[val_idx], preds))
    return np.mean(rmses)


def main():
    l1 = "es"
    print(f"Loading {l1.upper()} data...")
    train_df = pd.read_csv(DATA_DIR / "train" / l1 / f"kvl_shared_task_{l1}_train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev" / l1 / f"kvl_shared_task_{l1}_dev.csv")

    # Base features
    X_train_base = build_features(train_df, BASE_FEATURES)
    X_dev_base = build_features(dev_df, BASE_FEATURES)
    y_train = train_df["GLMM_score"].values
    y_dev = dev_df["GLMM_score"].values

    # Baseline CV score
    base_cv = cv_rmse(X_train_base, y_train)

    # Baseline dev score
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train_base, y_train, verbose=False)
    base_dev = root_mean_squared_error(y_dev, model.predict(X_dev_base))

    print(f"\nBaseline (full_xgb):  CV RMSE = {base_cv:.4f}  |  Dev RMSE = {base_dev:.4f}")
    print(f"{'='*70}")

    # ── Test each new feature individually ──
    NEW_FEATURES = [
        "suffix_transform_match",
        "is_multiword_source",
        "source_word_len",
        "len_ratio",
        "len_diff",
        "source_has_accent",
        "shared_suffix_ratio",
        "is_latinate",
        "syllable_count",
        "unique_char_ratio",
    ]

    results = []
    print(f"\n{'Feature':<28} {'CV RMSE':>8} {'Delta':>8} {'Dev RMSE':>9} {'Delta':>8}")
    print("-" * 65)

    for feat_name in NEW_FEATURES:
        new_feats_train = compute_new_features(train_df, [feat_name], l1=l1)
        new_feats_dev = compute_new_features(dev_df, [feat_name], l1=l1)

        X_train = pd.concat([X_train_base, new_feats_train], axis=1)
        X_dev = pd.concat([X_dev_base, new_feats_dev], axis=1)

        feat_cv = cv_rmse(X_train, y_train)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        feat_dev = root_mean_squared_error(y_dev, model.predict(X_dev))

        cv_delta = feat_cv - base_cv
        dev_delta = feat_dev - base_dev
        marker = " <--" if cv_delta < -0.002 else (" !!!" if cv_delta > 0.002 else "")

        print(f"{feat_name:<28} {feat_cv:>8.4f} {cv_delta:>+8.4f} {feat_dev:>9.4f} {dev_delta:>+8.4f}{marker}")
        results.append({"feature": feat_name, "cv_rmse": feat_cv, "cv_delta": cv_delta,
                        "dev_rmse": feat_dev, "dev_delta": dev_delta})

    # ── Combine all features with negative CV delta ──
    winners = [r["feature"] for r in results if r["cv_delta"] < 0]
    if winners:
        print(f"\n{'='*70}")
        print(f"Combined winners: {', '.join(winners)}")

        all_new_train = compute_new_features(train_df, winners, l1=l1)
        all_new_dev = compute_new_features(dev_df, winners, l1=l1)
        X_train_all = pd.concat([X_train_base, all_new_train], axis=1)
        X_dev_all = pd.concat([X_dev_base, all_new_dev], axis=1)

        combo_cv = cv_rmse(X_train_all, y_train)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train_all, y_train, verbose=False)
        combo_dev = root_mean_squared_error(y_dev, model.predict(X_dev_all))

        print(f"Combined CV RMSE:  {combo_cv:.4f}  (delta {combo_cv - base_cv:+.4f})")
        print(f"Combined Dev RMSE: {combo_dev:.4f}  (delta {combo_dev - base_dev:+.4f})")

        # Feature importance for combined model
        imp = model.feature_importances_
        feat_names = X_train_all.columns
        print(f"\nTop 15 feature importances:")
        for idx in np.argsort(imp)[::-1][:15]:
            print(f"  {feat_names[idx]:<28} {imp[idx]:.4f}")

    # ── Seed ensemble (5 seeds) on best feature set ──
    print(f"\n{'='*70}")
    print("Seed ensemble (5 seeds) on combined winners")

    if winners:
        X_tr = X_train_all
        X_dv = X_dev_all
    else:
        X_tr = X_train_base
        X_dv = X_dev_base

    ensemble_preds = []
    for seed in [42, 123, 456, 789, 1024]:
        params = {**XGB_PARAMS, "random_state": seed}
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_train, verbose=False)
        ensemble_preds.append(m.predict(X_dv))

    avg_preds = np.mean(ensemble_preds, axis=0)
    ens_dev = root_mean_squared_error(y_dev, avg_preds)
    ens_r, _ = pearsonr(avg_preds, y_dev)
    print(f"Ensemble Dev RMSE: {ens_dev:.4f}  (delta {ens_dev - base_dev:+.4f})")
    print(f"Ensemble Dev Pearson: {ens_r:.4f}")


if __name__ == "__main__":
    main()
