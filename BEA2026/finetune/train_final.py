"""
Final mDeBERTa fine-tuning for BEA2026 submission.

Trains on train+dev combined, predicts on test set.
Uses the embedding-enriched input format (with esim from LaBSE).

Usage (Colab):
    1. Upload vocab-difficulty/data/ to Google Drive under BEA2026/data/
    2. Run all cells — script auto-detects completed runs and skips them
    3. Final ensemble predictions are saved to Drive

Usage (local):
    python train_final.py --data_dir ../vocab-difficulty/data --output_dir ./output_final --no_drive
"""

import argparse
import logging
import os
import gc
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ── Configuration ──────────────────────────────────────────────────────────

MODEL_NAME = "microsoft/mdeberta-v3-base"
L1S = ["es"]
SEEDS = [10, 42, 123]
MAX_LENGTH = 256

TRAIN_ARGS = {
    "learning_rate": 2e-5,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "rmse",
    "greater_is_better": False,
    "logging_strategy": "epoch",
    "report_to": "none",
    "fp16": torch.cuda.is_available(),
}


# ── Embedding model (lazy-loaded) ────────────────────────────────────────

_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logging.getLogger("bea2026").info("Loading LaBSE embedding model...")
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/LaBSE")
    return _EMBED_MODEL


def compute_embed_cosine(en_words: list[str], l1_words: list[str]) -> np.ndarray:
    """Compute cosine similarity between LaBSE embeddings of EN and L1 words."""
    from sklearn.metrics.pairwise import cosine_similarity
    model = _get_embed_model()
    en_embs = model.encode(en_words, batch_size=256, show_progress_bar=False)
    l1_embs = model.encode(l1_words, batch_size=256, show_progress_bar=False)
    return np.array([
        cosine_similarity([e], [l])[0, 0] for e, l in zip(en_embs, l1_embs)
    ])


# ── Feature computation ───────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    """Space-optimized Levenshtein distance."""
    a, b = a.lower(), b.lower()
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_clue_ratio(clue: str, word: str) -> float:
    """Ratio of revealed (non-underscore) characters in clue to word length."""
    revealed = sum(1 for c in clue if c != "_")
    return round(revealed / max(len(word), 1), 2)


def build_enriched_input(row, sep_token: str = "[SEP]", embed_cosine: float = None) -> str:
    """Build feature-enriched input string for a single row."""
    en_word = str(row["en_target_word"]).strip()
    l1_word = str(row["L1_source_word"]).strip()
    pos = str(row["en_target_pos"]).strip()
    clue = str(row["en_target_clue"]).strip()
    context = str(row["L1_context"]).strip()

    wlen = len(en_word)
    nedit = round(levenshtein(en_word, l1_word) / max(len(en_word), len(l1_word), 1), 2)
    clue_ratio = compute_clue_ratio(clue, en_word)

    feat_str = f"wlen={wlen} | nedit={nedit} | pos={pos} | clue={clue_ratio}"
    if embed_cosine is not None:
        feat_str += f" | esim={embed_cosine:.3f}"

    sep = f" {sep_token} "
    text_str = sep.join([l1_word, context, clue, en_word])

    return f"{feat_str} | {text_str}"


# ── Data loading ──────────────────────────────────────────────────────────

def load_split(data_dir: Path, split: str, l1: str) -> pd.DataFrame:
    """Load a single data split CSV."""
    path = data_dir / split / l1 / f"kvl_shared_task_{l1}_{split}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def prepare_dataset(df: pd.DataFrame, sep_token: str = "[SEP]", target_mean: float = None, target_std: float = None):
    """Build enriched input strings and optionally scale targets."""
    en_words = df["en_target_word"].astype(str).tolist()
    l1_words = df["L1_source_word"].astype(str).tolist()
    cosine_sims = compute_embed_cosine(en_words, l1_words)

    inputs = [
        build_enriched_input(row, sep_token=sep_token, embed_cosine=cosine_sims[i])
        for i, (_, row) in enumerate(df.iterrows())
    ]
    item_ids = df["item_id"].tolist()

    data = {"input_text": inputs, "item_id": item_ids}

    if "GLMM_score" in df.columns:
        labels = df["GLMM_score"].values.astype(np.float32)
        if target_mean is None:
            target_mean = float(labels.mean())
            target_std = float(labels.std())
        scaled_labels = ((labels - target_mean) / target_std).tolist()
        data["label"] = scaled_labels

    ds = Dataset.from_dict(data)
    return ds, target_mean, target_std


# ── Metrics ───────────────────────────────────────────────────────────────

def make_compute_metrics(target_mean: float, target_std: float):
    """Create a compute_metrics function that inverse-scales predictions."""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        preds_orig = predictions * target_std + target_mean
        labels_orig = labels * target_std + target_mean
        rmse = root_mean_squared_error(labels_orig, preds_orig)
        r, _ = pearsonr(preds_orig, labels_orig)
        return {"rmse": rmse, "pearson": r}
    return compute_metrics


# ── Training ──────────────────────────────────────────────────────────────

def train_single(
    l1: str,
    seed: int,
    train_ds: Dataset,
    val_ds: Dataset,
    target_mean: float,
    target_std: float,
    output_dir: Path,
    tokenizer,
):
    """Fine-tune a single model for one (L1, seed) combination."""
    run_name = f"mdeberta_{l1}_seed{seed}"
    run_dir = output_dir / run_name

    # Check if already completed
    if (run_dir / "model.safetensors").exists() or (run_dir / "pytorch_model.bin").exists():
        logging.getLogger("bea2026").info(f"  [SKIP] {run_name} — checkpoint already exists")
        return run_dir

    # Check for intermediate checkpoint to resume from
    resume_from = None
    if run_dir.exists():
        checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            logging.getLogger("bea2026").info(f"  [RESUME] {run_name} from {checkpoints[-1].name}")
        else:
            logging.getLogger("bea2026").info(f"  [TRAIN] {run_name}")
    else:
        logging.getLogger("bea2026").info(f"  [TRAIN] {run_name}")

    def tokenize(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["input_text", "item_id"])
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=["input_text", "item_id"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model = model.float()

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        seed=seed,
        data_seed=seed,
        **TRAIN_ARGS,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_compute_metrics(target_mean, target_std),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except Exception as e:
        if resume_from:
            logging.getLogger("bea2026").info(f"  [WARN] Resume failed ({e}), retraining from scratch")
            import shutil
            for cp in run_dir.glob("checkpoint-*"):
                shutil.rmtree(cp)
            trainer.train()
        else:
            raise
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    del model, trainer, train_tok, val_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_dir


# ── Prediction ────────────────────────────────────────────────────────────

def predict_single(model_dir: Path, dataset: Dataset, tokenizer) -> np.ndarray:
    """Run inference with a saved model checkpoint."""
    def tokenize(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    ds_tok = dataset.map(tokenize, batched=True, remove_columns=["input_text", "item_id"])
    if "label" in ds_tok.column_names:
        ds_tok = ds_tok.remove_columns(["label"])

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    preds = trainer.predict(ds_tok).predictions.flatten()

    del model, trainer, ds_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BEA2026 mDeBERTa final training (train+dev → test)")
    parser.add_argument("--data_dir", type=str, default="../vocab-difficulty/data",
                        help="Path to data directory containing train/dev/test splits")
    parser.add_argument("--output_dir", type=str, default="./output_final",
                        help="Path to save checkpoints and predictions")
    parser.add_argument("--no_drive", action="store_true",
                        help="Skip Google Drive mounting (for local runs)")
    args = parser.parse_args()

    # Google Drive setup
    if not args.no_drive:
        drive_path = Path("/content/drive/MyDrive")
        if drive_path.exists():
            drive_base = drive_path / "BEA2026"
            drive_base.mkdir(parents=True, exist_ok=True)
            data_candidate = drive_base / "data"
            if (data_candidate / "data" / "train").exists():
                data_candidate = data_candidate / "data"
            args.data_dir = str(data_candidate)
            args.output_dir = str(drive_base / "output_final")
            print(f"Google Drive detected. Data: {args.data_dir}, Output: {args.output_dir}")
        else:
            print("Drive not mounted. Run this in a Colab cell first:")
            print("  from google.colab import drive; drive.mount('/content/drive')")
            print("Or use: python train_final.py --no_drive --data_dir /path/to/data")
            return

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_path = output_dir / f"train_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log = logging.getLogger("bea2026")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(sh)
    log.info(f"Logging to {log_path}")
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sep_token = tokenizer.sep_token

    # ── Train on train+dev, use dev as eval for early stopping ─────────
    log.info("\n=== FINAL TRAINING (train+dev) ===\n")

    scaling_params = {}
    datasets_cache = {}

    for l1 in L1S:
        log.info(f"\n--- L1: {l1.upper()} ---")

        train_df = load_split(data_dir, "train", l1)
        dev_df = load_split(data_dir, "dev", l1)

        if train_df is None or dev_df is None:
            log.info(f"  [ERROR] Missing data for {l1}, skipping")
            continue

        # Combine train + dev for training
        full_df = pd.concat([train_df, dev_df], ignore_index=True)
        log.info(f"  Combined: {len(train_df)} train + {len(dev_df)} dev = {len(full_df)} total")

        # Scaling from full dataset
        full_ds, t_mean, t_std = prepare_dataset(full_df, sep_token=sep_token)
        # Use dev portion as eval set for early stopping
        val_ds, _, _ = prepare_dataset(dev_df, sep_token=sep_token, target_mean=t_mean, target_std=t_std)
        scaling_params[l1] = (t_mean, t_std)
        datasets_cache[l1] = full_df

        log.info(f"  Target scaling: mean={t_mean:.4f}, std={t_std:.4f}")

        for seed in SEEDS:
            train_single(l1, seed, full_ds, val_ds, t_mean, t_std, output_dir, tokenizer)

    # ── Generate test predictions ──────────────────────────────────────
    log.info("\n=== TEST PREDICTIONS ===\n")

    for l1 in L1S:
        if l1 not in scaling_params:
            continue

        t_mean, t_std = scaling_params[l1]

        test_df = load_split(data_dir, "test", l1)
        if test_df is None:
            log.info(f"  [SKIP] No test data for {l1}")
            continue

        test_ds, _, _ = prepare_dataset(test_df, sep_token=sep_token, target_mean=t_mean, target_std=t_std)
        item_ids = test_df["item_id"].values
        all_preds = []

        for seed in SEEDS:
            run_name = f"mdeberta_{l1}_seed{seed}"
            model_dir = output_dir / run_name

            if not model_dir.exists():
                log.info(f"  [SKIP] {run_name} — no checkpoint found")
                continue

            log.info(f"  [PREDICT] {run_name} on test")
            raw_preds = predict_single(model_dir, test_ds, tokenizer)
            preds = raw_preds * t_std + t_mean
            all_preds.append(preds)

            # Save per-seed predictions
            seed_dir = pred_dir / "per_seed" / "test" / l1
            seed_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "item_id": item_ids,
                "prediction": preds,
            }).sort_values("item_id").to_csv(
                seed_dir / f"mdeberta_seed{seed}_preds.csv", index=False
            )

        if not all_preds:
            log.info(f"  [ERROR] No predictions for {l1}")
            continue

        # Ensemble average
        ensemble_preds = np.mean(all_preds, axis=0)

        # Save ensemble in submission format
        ens_dir = pred_dir / "closed" / "test" / l1
        ens_dir.mkdir(parents=True, exist_ok=True)
        ens_path = ens_dir / "mdeberta_embed_ensemble_preds.csv"
        pd.DataFrame({
            "item_id": item_ids,
            "prediction": ensemble_preds,
        }).sort_values("item_id").to_csv(ens_path, index=False)

        log.info(f"  Saved: {ens_path}")
        log.info(f"  Pred range: [{ensemble_preds.min():.3f}, {ensemble_preds.max():.3f}], mean={ensemble_preds.mean():.3f}")

        # Per-seed stats
        for i, seed in enumerate(SEEDS):
            p = all_preds[i]
            log.info(f"    seed {seed}: range=[{p.min():.3f}, {p.max():.3f}], mean={p.mean():.3f}")

    log.info("\n=== DONE ===")


if __name__ == "__main__":
    main()
