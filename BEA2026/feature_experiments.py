"""
Feature experiments for vocabulary difficulty prediction.
Tries combinations of simple features with linear/ridge/xgboost regression.

Results are cached in results_cache.csv — only new experiments are run.
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr
import jellyfish
from wordfreq import zipf_frequency
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ── Lazy-loaded embedding model (cached across calls) ──
_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print("Loading multilingual embedding model (LaBSE)...")
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/LaBSE")
    return _EMBED_MODEL

DATA_DIR = Path("vocab-difficulty/data")
CACHE_PATH = Path("results/results_cache.csv")
PRED_DIR = Path("predictions")
L1S = ["es", "de", "cn"]

# ── Suffix / prefix lists for one-hot features ──
TOP_SUFFIXES = ["-tion", "-sion", "-ing", "-ence", "-ment", "-able", "-ness", "-ly"]
TOP_PREFIXES = ["dis-", "re-", "un-", "sub-", "out-", "over-"]

# ── L1-specific suffix transformation rules ──
ES_SUFFIX_RULES = [
    ("tion", "ción"), ("sion", "sión"), ("ty", "dad"), ("ism", "ismo"),
    ("ist", "ista"), ("al", "al"), ("ble", "ble"), ("ous", "oso"),
    ("ive", "ivo"), ("ence", "encia"), ("ance", "ancia"), ("ly", "mente"),
    ("ment", "mento"), ("ant", "ante"), ("ent", "ente"),
]
DE_SUFFIX_RULES = [
    ("tion", "tion"), ("tion", "ierung"), ("sion", "sion"),
    ("ty", "tät"), ("ism", "ismus"), ("ist", "ist"),
    ("al", "al"), ("al", "ell"), ("ble", "bel"),
    ("ous", "ös"), ("ive", "iv"), ("ence", "enz"),
    ("ance", "anz"), ("ment", "ment"), ("ant", "ant"),
    ("ent", "ent"), ("ic", "isch"), ("ly", "lich"),
]
SUFFIX_RULES_BY_L1 = {"es": ES_SUFFIX_RULES, "de": DE_SUFFIX_RULES}


def edit_distance(a, b):
    """Levenshtein distance between two strings."""
    a, b = a.lower(), b.lower()
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m]


def count_vowels(word):
    return sum(1 for c in word.lower() if c in 'aeiou')


def count_consonants(word):
    return sum(1 for c in word.lower() if c.isalpha() and c not in 'aeiou')


def shared_prefix_ratio(en, l1):
    """Ratio of shared leading characters to max word length."""
    en, l1 = en.lower(), l1.lower()
    shared = 0
    for a, b in zip(en, l1):
        if a == b:
            shared += 1
        else:
            break
    return shared / max(len(en), len(l1), 1)


def max_consonant_cluster(word):
    """Length of the longest consecutive consonant sequence."""
    clusters = re.findall(r'[^aeiou\s]+', word.lower())
    return max((len(c) for c in clusters), default=0)


def has_double_letter(word):
    """Binary: does the word contain consecutive identical letters."""
    w = word.lower()
    return int(any(w[i] == w[i+1] for i in range(len(w) - 1)))


def char_ngram_overlap(en, l1, n=2):
    """Jaccard similarity of character n-grams between two words."""
    en, l1 = en.lower(), l1.lower()
    en_ngrams = set(en[i:i+n] for i in range(len(en) - n + 1))
    l1_ngrams = set(l1[i:i+n] for i in range(len(l1) - n + 1))
    if not en_ngrams or not l1_ngrams:
        return 0.0
    return len(en_ngrams & l1_ngrams) / len(en_ngrams | l1_ngrams)


def shared_suffix_ratio_fn(en, l1):
    """Ratio of shared trailing characters to max word length."""
    en, l1 = en.lower()[::-1], l1.lower()[::-1]
    shared = 0
    for a, b in zip(en, l1):
        if a == b:
            shared += 1
        else:
            break
    return shared / max(len(en), len(l1), 1)


def count_syllables(word):
    word = word.lower()
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and not word.endswith('le'):
        count = max(1, count - 1)
    return max(1, count)


def count_de_syllables(word):
    count = len(re.findall(r'[aeiouyäöü]+', word.lower()))
    return max(1, count)


def build_features(df, feature_set, l1=None):
    feats = {}
    if "word_len" in feature_set:
        feats["word_len"] = df["en_target_word"].str.len()
    if "edit_dist" in feature_set:
        feats["edit_dist"] = df.apply(
            lambda r: edit_distance(r["en_target_word"], str(r["L1_source_word"])), axis=1)
    if "norm_edit_dist" in feature_set:
        feats["norm_edit_dist"] = df.apply(
            lambda r: edit_distance(r["en_target_word"], str(r["L1_source_word"]))
                      / max(len(r["en_target_word"]), len(str(r["L1_source_word"])), 1), axis=1)
    if "vowels" in feature_set:
        feats["vowels"] = df["en_target_word"].apply(count_vowels)
    if "consonants" in feature_set:
        feats["consonants"] = df["en_target_word"].apply(count_consonants)
    if "vowel_ratio" in feature_set:
        feats["vowel_ratio"] = df["en_target_word"].apply(
            lambda w: count_vowels(w) / max(len(w), 1))
    if "pos" in feature_set:
        ALL_POS = ["adjective", "adverb", "misc", "not-no", "noun", "number", "preposition", "verb"]
        for pos in ALL_POS:
            feats[f"pos_{pos}"] = (df["en_target_pos"] == pos).astype(int)
    # ── Tier 1 features ──
    if "suffix" in feature_set:
        for sfx in TOP_SUFFIXES:
            col = f"sfx_{sfx.strip('-')}"
            feats[col] = df["en_target_word"].str.lower().str.endswith(sfx.strip("-")).astype(int)
    if "prefix" in feature_set:
        for pfx in TOP_PREFIXES:
            col = f"pfx_{pfx.strip('-')}"
            feats[col] = df["en_target_word"].str.lower().str.startswith(pfx.strip("-")).astype(int)
    if "shared_prefix_ratio" in feature_set:
        feats["shared_prefix_ratio"] = df.apply(
            lambda r: shared_prefix_ratio(r["en_target_word"], str(r["L1_source_word"])), axis=1)
    if "context_len" in feature_set:
        feats["context_len"] = df["L1_context"].astype(str).str.len()
    if "max_consonant_cluster" in feature_set:
        feats["max_consonant_cluster"] = df["en_target_word"].apply(max_consonant_cluster)
    if "double_letter" in feature_set:
        feats["double_letter"] = df["en_target_word"].apply(has_double_letter)
    # ── Tier 3 features ──
    if "jaro_winkler" in feature_set:
        feats["jaro_winkler"] = df.apply(
            lambda r: jellyfish.jaro_winkler_similarity(
                r["en_target_word"].lower(), str(r["L1_source_word"]).lower()), axis=1)
    if "ngram_overlap" in feature_set:
        feats["ngram_overlap"] = df.apply(
            lambda r: char_ngram_overlap(r["en_target_word"], str(r["L1_source_word"])), axis=1)
    if "first_letter_match" in feature_set:
        feats["first_letter_match"] = df.apply(
            lambda r: int(r["en_target_word"][0].lower() == str(r["L1_source_word"])[0].lower())
            if len(r["en_target_word"]) > 0 and len(str(r["L1_source_word"])) > 0 else 0, axis=1)
    if "zipf_freq" in feature_set:
        feats["zipf_freq"] = df["en_target_word"].apply(
            lambda w: zipf_frequency(w.lower(), "en"))
    # ── V2 features (from ES/DE ablation studies) ──
    en = df["en_target_word"].str.lower()
    src = df["L1_source_word"].astype(str).str.lower()
    if "suffix_transform_match" in feature_set:
        rules = SUFFIX_RULES_BY_L1.get(l1, [])
        def _check(row):
            e, s = row["en_target_word"].lower(), str(row["L1_source_word"]).lower()
            return int(any(e.endswith(es) and s.endswith(ls) for es, ls in rules))
        feats["suffix_transform_match"] = df.apply(_check, axis=1)
    if "is_multiword_source" in feature_set:
        feats["is_multiword_source"] = src.str.contains(r'\s').astype(int)
    if "source_word_len" in feature_set:
        feats["source_word_len"] = src.str.len()
    if "len_ratio" in feature_set:
        feats["len_ratio"] = src.str.len() / en.str.len().clip(lower=1)
    if "len_diff" in feature_set:
        feats["len_diff"] = src.str.len() - en.str.len()
    if "source_has_accent" in feature_set:
        feats["source_has_accent"] = src.str.contains(r'[áéíóúàèìòùäëïöüß]').astype(int)
    if "shared_suffix_ratio" in feature_set:
        feats["shared_suffix_ratio"] = df.apply(
            lambda r: shared_suffix_ratio_fn(r["en_target_word"], str(r["L1_source_word"])), axis=1)
    if "is_latinate" in feature_set:
        _lat = re.compile(r'(tion|ment|ence|ance|ible|able|ious|eous|ous|ive|ant|ent|ism|ist|ure|ary|ory)')
        feats["is_latinate"] = en.apply(lambda w: int(bool(_lat.search(w))))
    if "syllable_count" in feature_set:
        feats["syllable_count"] = en.apply(count_syllables)
    if "source_syllable_count" in feature_set:
        feats["source_syllable_count"] = src.apply(count_de_syllables)
    # ── Cross-lingual embedding cosine similarity ──
    if "embed_cosine" in feature_set:
        model = _get_embed_model()
        en_words = df["en_target_word"].tolist()
        l1_words = df["L1_source_word"].astype(str).tolist()
        en_embs = model.encode(en_words, batch_size=256, show_progress_bar=False)
        l1_embs = model.encode(l1_words, batch_size=256, show_progress_bar=False)
        # Row-wise cosine similarity
        sims = np.array([
            sklearn_cosine([e], [l])[0, 0] for e, l in zip(en_embs, l1_embs)
        ])
        feats["embed_cosine"] = sims
    return pd.DataFrame(feats)


class ThresholdModel:
    """If edit distance is small (cognate), use edit_dist features; otherwise use word_len."""
    def __init__(self, threshold):
        self.threshold = threshold
        self.model_close = LinearRegression()
        self.model_far = LinearRegression()

    def fit(self, X_df, y):
        close = X_df["norm_edit_dist"] <= self.threshold
        if close.sum() > 5:
            self.model_close.fit(X_df.loc[close, ["norm_edit_dist", "edit_dist"]], y[close])
        if (~close).sum() > 5:
            self.model_far.fit(X_df.loc[~close, ["word_len"]], y[~close])
        return self

    def predict(self, X_df):
        preds = np.zeros(len(X_df))
        close = X_df["norm_edit_dist"] <= self.threshold
        if close.sum() > 0:
            preds[close] = self.model_close.predict(X_df.loc[close, ["norm_edit_dist", "edit_dist"]])
        if (~close).sum() > 0:
            preds[~close] = self.model_far.predict(X_df.loc[~close, ["word_len"]])
        return preds


# ── Load cached results ──
if CACHE_PATH.exists():
    cached = pd.read_csv(CACHE_PATH)
else:
    cached = pd.DataFrame(columns=["experiment", "L1", "RMSE", "Pearson"])


# ── Experiment definitions ──
EXPERIMENTS = {
    "word_len (baseline)":       {"features": ["word_len"], "model": "lr"},
    "edit_dist":                 {"features": ["edit_dist"], "model": "lr"},
    "norm_edit_dist":            {"features": ["norm_edit_dist"], "model": "lr"},
    "word_len + edit_dist":      {"features": ["word_len", "edit_dist"], "model": "lr"},
    "word_len + norm_edit":      {"features": ["word_len", "norm_edit_dist"], "model": "lr"},
    "vowels":                    {"features": ["vowels"], "model": "lr"},
    "consonants":                {"features": ["consonants"], "model": "lr"},
    "vowels + consonants":       {"features": ["vowels", "consonants"], "model": "lr"},
    "vowel_ratio":               {"features": ["vowel_ratio"], "model": "lr"},
    "all_simple":                {"features": ["word_len", "norm_edit_dist", "vowels", "consonants"], "model": "lr"},
    "all_with_ratio":            {"features": ["word_len", "norm_edit_dist", "vowels", "consonants", "vowel_ratio"], "model": "lr"},
    "threshold_0.5":             {"features": ["word_len", "edit_dist", "norm_edit_dist"], "model": "threshold", "threshold": 0.5},
    "word_len + norm_edit + pos": {"features": ["word_len", "norm_edit_dist", "pos"], "model": "lr"},
    "best + pos":                {"features": ["word_len", "norm_edit_dist", "vowels", "consonants", "pos"], "model": "lr"},
    # ── Tier 1 + Ridge ──
    "tier1_ridge": {
        "features": ["word_len", "norm_edit_dist", "pos", "suffix", "prefix",
                      "shared_prefix_ratio", "context_len", "max_consonant_cluster", "double_letter"],
        "model": "ridge",
    },
    # ── Tier 1 + XGBoost ──
    "tier1_xgb": {
        "features": ["word_len", "norm_edit_dist", "pos", "suffix", "prefix",
                      "shared_prefix_ratio", "context_len", "max_consonant_cluster", "double_letter"],
        "model": "xgb",
    },
    # ── Full (Tier 1 + Tier 3) + XGBoost ──
    "full_xgb": {
        "features": ["word_len", "norm_edit_dist", "pos", "suffix", "prefix",
                      "shared_prefix_ratio", "context_len", "max_consonant_cluster", "double_letter",
                      "jaro_winkler", "ngram_overlap", "first_letter_match", "zipf_freq"],
        "model": "xgb",
    },
    # ── Full V2: all ablation winners + L1-tuned XGBoost ──
    "full_xgb_v2": {
        "features": ["word_len", "norm_edit_dist", "pos", "suffix", "prefix",
                      "shared_prefix_ratio", "context_len", "max_consonant_cluster", "double_letter",
                      "jaro_winkler", "ngram_overlap", "first_letter_match", "zipf_freq",
                      "suffix_transform_match", "is_multiword_source", "source_word_len",
                      "len_ratio", "len_diff", "source_has_accent", "shared_suffix_ratio",
                      "is_latinate", "syllable_count", "source_syllable_count"],
        "model": "xgb_tuned",
    },
    # ── Full XGBoost + cross-lingual embedding cosine similarity (LaBSE) ──
    "full_xgb_embed": {
        "features": ["word_len", "norm_edit_dist", "pos", "suffix", "prefix",
                      "shared_prefix_ratio", "context_len", "max_consonant_cluster", "double_letter",
                      "jaro_winkler", "ngram_overlap", "first_letter_match", "zipf_freq",
                      "embed_cosine"],
        "model": "xgb",
    },
    # ── Full V2 + embed cosine with tuned XGBoost ──
    "full_xgb_v2_embed": {
        "features": ["word_len", "norm_edit_dist", "pos", "suffix", "prefix",
                      "shared_prefix_ratio", "context_len", "max_consonant_cluster", "double_letter",
                      "jaro_winkler", "ngram_overlap", "first_letter_match", "zipf_freq",
                      "suffix_transform_match", "is_multiword_source", "source_word_len",
                      "len_ratio", "len_diff", "source_has_accent", "shared_suffix_ratio",
                      "is_latinate", "syllable_count", "source_syllable_count",
                      "embed_cosine"],
        "model": "xgb_tuned",
    },
}

# ── L1-specific tuned XGBoost params ──
XGB_TUNED_PARAMS = {
    "es": {
        "n_estimators": 600, "max_depth": 3, "learning_rate": 0.02,
        "subsample": 0.85, "colsample_bytree": 0.7, "reg_alpha": 0.5,
        "reg_lambda": 2.0, "random_state": 42, "verbosity": 0,
    },
    "de": {
        "n_estimators": 400, "max_depth": 3, "learning_rate": 0.05,
        "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1,
        "reg_lambda": 2.0, "random_state": 42, "verbosity": 0,
    },
    "cn": {
        "n_estimators": 400, "max_depth": 3, "learning_rate": 0.05,
        "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1,
        "reg_lambda": 2.0, "random_state": 42, "verbosity": 0,
    },
}

# ── Find which (experiment, L1) pairs need to run ──
cached_pairs = set()
if len(cached) > 0:
    cached_pairs = set(zip(cached["experiment"], cached["L1"]))
runs_needed = []
for exp_name, cfg in EXPERIMENTS.items():
    for l1 in L1S:
        if (exp_name, l1) not in cached_pairs:
            runs_needed.append((exp_name, l1, cfg))

if not runs_needed:
    print("No new experiments to run.")
else:
    exp_names_needed = sorted(set(e for e, _, _ in runs_needed))
    l1s_needed = sorted(set(l for _, l, _ in runs_needed))
    print(f"Running {len(runs_needed)} new (experiment, L1) pair(s) for: {', '.join(exp_names_needed)} × {', '.join(l1s_needed)}")

# ── Load data (only if there's something to run) ──
data = {}
if runs_needed:
    for l1 in l1s_needed:
        data[l1] = {
            "train": pd.read_csv(DATA_DIR / "train" / l1 / f"kvl_shared_task_{l1}_train.csv"),
            "dev": pd.read_csv(DATA_DIR / "dev" / l1 / f"kvl_shared_task_{l1}_dev.csv"),
        }

new_results = []
best_preds = {}  # track predictions for saving

for exp_name, l1, cfg in runs_needed:
    train_df = data[l1]["train"]
    dev_df = data[l1]["dev"]

    X_train = build_features(train_df, cfg["features"], l1=l1)
    y_train = train_df["GLMM_score"].values
    X_dev = build_features(dev_df, cfg["features"], l1=l1)
    y_dev = dev_df["GLMM_score"].values

    if cfg["model"] == "lr":
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)
    elif cfg["model"] == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)
    elif cfg["model"] == "xgb":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)
        preds = model.predict(X_dev)
    elif cfg["model"] == "xgb_tuned":
        import xgboost as xgb
        params = XGB_TUNED_PARAMS[l1]
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_dev)
    elif cfg["model"] == "threshold":
        model = ThresholdModel(cfg["threshold"])
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)

    rmse = root_mean_squared_error(y_dev, preds)
    r, _ = pearsonr(preds, y_dev)
    new_results.append({"experiment": exp_name, "L1": l1, "RMSE": rmse, "Pearson": r})
    best_preds[(exp_name, l1)] = {"preds": preds, "item_ids": dev_df["item_id"].values}

# ── Merge and save cache ──
if new_results:
    new_df = pd.DataFrame(new_results)
    cached = pd.concat([cached, new_df], ignore_index=True)
    cached.to_csv(CACHE_PATH, index=False)
    print(f"Cache updated: {CACHE_PATH}")

# ── Print all results ──
baseline_rmse = {"es": 1.357, "de": 1.328, "cn": 1.175}

all_exp_names = list(dict.fromkeys(cached["experiment"]))  # preserve order

# Print header based on which L1s are present
header = f"{'Experiment':<28}"
for l1 in L1S:
    header += f" {l1.upper()+' RMSE':>8}"
header += f" {'Avg RMSE':>9}"
for l1 in L1S:
    header += f" {l1.upper()+' r':>6}"
print(f"\n{header}")
print("-" * (28 + 9*len(L1S) + 10 + 7*len(L1S)))

for exp_name in all_exp_names:
    sub = cached[cached["experiment"] == exp_name]
    vals = {r["L1"]: r for _, r in sub.iterrows()}
    available_l1s = [l1 for l1 in L1S if l1 in vals]
    if not available_l1s:
        continue
    avg_rmse = np.mean([vals[l1]["RMSE"] for l1 in available_l1s])
    row = f"{exp_name:<28}"
    for l1 in L1S:
        row += f" {vals[l1]['RMSE']:>8.3f}" if l1 in vals else f" {'—':>8}"
    row += f" {avg_rmse:>9.3f}"
    for l1 in L1S:
        row += f" {vals[l1]['Pearson']:>6.3f}" if l1 in vals else f" {'—':>6}"
    print(row)

bl_avg = np.mean([baseline_rmse[l1] for l1 in L1S])
bl_row = f"{'XLM-R baseline':<28}"
for l1 in L1S:
    bl_row += f" {baseline_rmse[l1]:>8.3f}"
bl_row += f" {bl_avg:>9.3f}"
print(bl_row)

# ── Save best model predictions ──
# Only consider experiments that have all L1s
full_exps = [e for e in all_exp_names
             if all(len(cached[(cached["experiment"] == e) & (cached["L1"] == l1)]) > 0 for l1 in L1S)]
best_exp = min(
    full_exps if full_exps else all_exp_names,
    key=lambda e: np.mean([
        cached[(cached["experiment"] == e) & (cached["L1"] == l1)]["RMSE"].values[0]
        for l1 in L1S
        if len(cached[(cached["experiment"] == e) & (cached["L1"] == l1)]) > 0
    ])
)
print(f"\nBest experiment: {best_exp}")

# Only save predictions if best experiment was just run (we have its preds in memory)
if any(best_exp == r["experiment"] for r in new_results):
    for l1 in L1S:
        bp = best_preds[(best_exp, l1)]
        out_dir = PRED_DIR / "closed" / "dev" / l1
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "feature_model_preds.csv"
        pd.DataFrame({"item_id": bp["item_ids"], "prediction": bp["preds"]}).to_csv(out_path, index=False)
        print(f"  Saved {out_path}")
else:
    print("  (predictions already saved from previous run)")
