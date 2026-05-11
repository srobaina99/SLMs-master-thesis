"""
Test DE-targeted features individually and in combination.
Same approach as test_es_features.py but with DE-specific adaptations.
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
from test_es_features import compute_new_features

BASE_FEATURES = [
    "word_len", "norm_edit_dist", "pos", "suffix", "prefix",
    "shared_prefix_ratio", "context_len", "max_consonant_cluster", "double_letter",
    "jaro_winkler", "ngram_overlap", "first_letter_match", "zipf_freq",
]

# DE-specific suffix transform rules (English → German)
DE_SUFFIX_RULES = [
    ("tion", "tion"), ("tion", "ierung"), ("sion", "sion"),
    ("ty", "tät"), ("ism", "ismus"), ("ist", "ist"),
    ("al", "al"), ("al", "ell"), ("ble", "bel"),
    ("ous", "ös"), ("ive", "iv"), ("ence", "enz"),
    ("ance", "anz"), ("ment", "ment"), ("ant", "ant"),
    ("ent", "ent"), ("ic", "isch"), ("ly", "lich"),
]

# ── Language-agnostic + DE-specific feature computation ──

def compute_de_features(df, feature_keys, l1="de"):
    """Compute features for DE. Reuses language-agnostic ones from test_es_features,
    overrides suffix_transform_match with DE rules, adds DE-specific features."""
    feats = {}
    en = df["en_target_word"].str.lower()
    src = df["L1_source_word"].astype(str).str.lower()

    # Language-agnostic features — delegate to ES version
    agnostic = ["is_multiword_source", "source_word_len", "len_ratio", "len_diff",
                "shared_suffix_ratio", "syllable_count", "unique_char_ratio"]
    for feat in agnostic:
        if feat in feature_keys:
            result = compute_new_features(df, [feat], l1=l1)
            for col in result.columns:
                feats[col] = result[col]

    # DE-specific suffix transform match
    if "suffix_transform_match" in feature_keys:
        def check_de_suffix(row):
            e = row["en_target_word"].lower()
            s = str(row["L1_source_word"]).lower()
            for en_sfx, de_sfx in DE_SUFFIX_RULES:
                if e.endswith(en_sfx) and s.endswith(de_sfx):
                    return 1
            return 0
        feats["suffix_transform_match"] = df.apply(check_de_suffix, axis=1)

    # DE: source has umlaut (ä, ö, ü, ß)
    if "source_has_accent" in feature_keys:
        feats["source_has_accent"] = src.str.contains(r'[äöüßÄÖÜ]').astype(int)

    # is_latinate — same logic, works for DE too (Latinate EN words may be harder for DE speakers)
    if "is_latinate" in feature_keys:
        latin_patterns = re.compile(
            r'(tion|ment|ence|ance|ible|able|ious|eous|ous|ive|ant|ent|ism|ist|ure|ary|ory)')
        feats["is_latinate"] = en.apply(lambda w: int(bool(latin_patterns.search(w))))

    # DE-specific: is source a compound (very long, no spaces)
    if "is_compound_source" in feature_keys:
        feats["is_compound_source"] = (src.str.len() > 12).astype(int) & (~src.str.contains(r'\s')).astype(int)

    # DE-specific: source word syllable estimate (German vowel groups)
    if "source_syllable_count" in feature_keys:
        def de_syllables(word):
            count = len(re.findall(r'[aeiouyäöü]+', word.lower()))
            return max(1, count)
        feats["source_syllable_count"] = src.apply(de_syllables)

    return pd.DataFrame(feats)


# ── XGBoost params (from tuning, DE-optimized) ──
XGB_PARAMS = {
    "n_estimators": 400, "max_depth": 3, "learning_rate": 0.05,
    "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1,
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
                       compute_de_features(train_df, new_feats)], axis=1)
    X_dv = pd.concat([build_features(dev_df, BASE_FEATURES),
                       compute_de_features(dev_df, new_feats)], axis=1)
    return X_tr, X_dv


def main():
    l1 = "de"
    print(f"Loading {l1.upper()} data...")
    train_df = pd.read_csv(DATA_DIR / "train" / l1 / f"kvl_shared_task_{l1}_train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev" / l1 / f"kvl_shared_task_{l1}_dev.csv")
    y_train = train_df["GLMM_score"].values
    y_dev = dev_df["GLMM_score"].values

    # Base features
    X_train_base = build_features(train_df, BASE_FEATURES)
    X_dev_base = build_features(dev_df, BASE_FEATURES)

    base_cv = cv_rmse(X_train_base, y_train)
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_train_base, y_train, verbose=False)
    base_dev = root_mean_squared_error(y_dev, m.predict(X_dev_base))

    print(f"\nBaseline (full_xgb):  CV RMSE = {base_cv:.4f}  |  Dev RMSE = {base_dev:.4f}")
    print(f"{'='*70}")

    # ── Test each feature individually ──
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
        "is_compound_source",
        "source_syllable_count",
    ]

    results = []
    print(f"\n{'Feature':<28} {'CV RMSE':>8} {'Delta':>8} {'Dev RMSE':>9} {'Delta':>8}")
    print("-" * 65)

    for feat_name in NEW_FEATURES:
        new_feats_train = compute_de_features(train_df, [feat_name])
        new_feats_dev = compute_de_features(dev_df, [feat_name])

        X_train = pd.concat([X_train_base, new_feats_train], axis=1)
        X_dev = pd.concat([X_dev_base, new_feats_dev], axis=1)

        feat_cv = cv_rmse(X_train, y_train)
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_train, y_train, verbose=False)
        feat_dev = root_mean_squared_error(y_dev, m.predict(X_dev))

        cv_d = feat_cv - base_cv
        dev_d = feat_dev - base_dev
        marker = " <--" if cv_d < -0.002 else (" !!!" if cv_d > 0.002 else "")
        print(f"{feat_name:<28} {feat_cv:>8.4f} {cv_d:>+8.4f} {feat_dev:>9.4f} {dev_d:>+8.4f}{marker}")
        results.append({"feature": feat_name, "cv_rmse": feat_cv, "cv_delta": cv_d,
                        "dev_rmse": feat_dev, "dev_delta": dev_d})

    # ── Combine winners ──
    winners = [r["feature"] for r in results if r["cv_delta"] < 0]
    if winners:
        print(f"\n{'='*70}")
        print(f"Combined winners ({len(winners)}): {', '.join(winners)}")
        X_tr, X_dv = build_full(train_df, dev_df, winners)
        combo_cv = cv_rmse(X_tr, y_train)
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr, y_train, verbose=False)
        combo_dev = root_mean_squared_error(y_dev, m.predict(X_dv))
        combo_r, _ = pearsonr(m.predict(X_dv), y_dev)
        print(f"Combined CV RMSE:  {combo_cv:.4f}  (delta {combo_cv - base_cv:+.4f})")
        print(f"Combined Dev RMSE: {combo_dev:.4f}  (delta {combo_dev - base_dev:+.4f})")
        print(f"Combined Dev Pearson: {combo_r:.4f}")

        # Drop-one ablation on winners
        print(f"\n{'Dropped feature':<28} {'CV RMSE':>8} {'Delta':>8} {'Dev RMSE':>9} {'Delta':>8}")
        print("-" * 65)
        for feat in winners:
            subset = [f for f in winners if f != feat]
            X_tr2, X_dv2 = build_full(train_df, dev_df, subset)
            ab_cv = cv_rmse(X_tr2, y_train)
            m2 = xgb.XGBRegressor(**XGB_PARAMS)
            m2.fit(X_tr2, y_train, verbose=False)
            ab_dev = root_mean_squared_error(y_dev, m2.predict(X_dv2))
            cv_d = ab_cv - combo_cv
            dev_d = ab_dev - combo_dev
            marker = " DROP" if cv_d < -0.0005 else (" KEEP" if cv_d > 0.001 else "")
            print(f"- {feat:<26} {ab_cv:>8.4f} {cv_d:>+8.4f} {ab_dev:>9.4f} {dev_d:>+8.4f}{marker}")

        # Feature importance
        print(f"\nTop 15 feature importances:")
        imp = m.feature_importances_
        feat_names = X_dv.columns
        for idx in np.argsort(imp)[::-1][:15]:
            print(f"  {feat_names[idx]:<28} {imp[idx]:.4f}")

    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Original full_xgb (no new feats):  Dev = {base_dev:.4f}")
    if winners:
        print(f"  Combined winners ({len(winners)} feats):       Dev = {combo_dev:.4f}")


if __name__ == "__main__":
    main()
