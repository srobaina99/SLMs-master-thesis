"""
Feature-enriched mDeBERTa fine-tuning for BEA2026 Vocabulary Difficulty Prediction.

Designed to run on Google Colab (free-tier T4 GPU).
Saves checkpoints to Google Drive for session resilience.

Usage (Colab):
    1. Upload vocab-difficulty/data/ to Google Drive under BEA2026/data/
    2. Run all cells — script auto-detects completed runs and skips them
    3. Final ensemble predictions are saved to Drive

Usage (local):
    python train.py --data_dir ../vocab-difficulty/data --output_dir ./output --no_drive
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
from datasets import Dataset, DatasetDict
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

    # Compute features
    wlen = len(en_word)
    nedit = round(levenshtein(en_word, l1_word) / max(len(en_word), len(l1_word), 1), 2)
    clue_ratio = compute_clue_ratio(clue, en_word)

    # Feature prefix
    feat_str = f"wlen={wlen} | nedit={nedit} | pos={pos} | clue={clue_ratio}"
    if embed_cosine is not None:
        feat_str += f" | esim={embed_cosine:.3f}"

    # Baseline-style text fields separated by model's sep token
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
    """
    Build enriched input strings and optionally scale targets.

    Returns:
        HF Dataset, target_mean, target_std
    """
    # Precompute embedding cosine similarities for the whole dataframe
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
        # Inverse-scale both to original GLMM space
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
    model = model.float()  # Ensure fp32 weights before fp16 AMP training

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
            # Clean up incompatible checkpoints
            import shutil
            for cp in run_dir.glob("checkpoint-*"):
                shutil.rmtree(cp)
            trainer.train()
        else:
            raise
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    # Cleanup
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
    parser = argparse.ArgumentParser(description="BEA2026 mDeBERTa fine-tuning")
    parser.add_argument("--data_dir", type=str, default="../vocab-difficulty/data",
                        help="Path to data directory containing train/dev/test splits")
    parser.add_argument("--output_dir", type=str, default="./output_embedding",
                        help="Path to save checkpoints and predictions")
    parser.add_argument("--no_drive", action="store_true",
                        help="Skip Google Drive mounting (for local runs)")
    parser.add_argument("--predict_split", type=str, default="dev",
                        choices=["dev", "test", "both"],
                        help="Which split(s) to generate predictions for")
    args = parser.parse_args()

    # Google Drive setup
    if not args.no_drive:
        drive_path = Path("/content/drive/MyDrive")
        if drive_path.exists():
            # Drive already mounted (e.g., via notebook cell)
            drive_base = drive_path / "BEA2026"
            drive_base.mkdir(parents=True, exist_ok=True)
            # Auto-detect double nesting (data/ vs data/data/)
            data_candidate = drive_base / "data"
            if (data_candidate / "data" / "train").exists():
                data_candidate = data_candidate / "data"
            args.data_dir = str(data_candidate)
            args.output_dir = str(drive_base / "output_embedding")
            print(f"Google Drive detected. Data: {args.data_dir}, Output: {args.output_dir}")
        else:
            print("Drive not mounted. Run this in a Colab cell first:")
            print("  from google.colab import drive; drive.mount('/content/drive')")
            print("Or use: python train.py --no_drive --data_dir /path/to/data")
            return

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to file + console
    log_path = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sep_token = tokenizer.sep_token  # [SEP] for mDeBERTa, </s> for XLM-R

    # ── Train all (L1, seed) combinations ─────────────────────────────
    log.info("\n=== TRAINING ===\n")

    scaling_params = {}  # l1 -> (mean, std)
    datasets_cache = {}  # l1 -> (train_ds, val_ds)

    for l1 in L1S:
        log.info(f"\n--- L1: {l1.upper()} ---")

        train_df = load_split(data_dir, "train", l1)
        dev_df = load_split(data_dir, "dev", l1)

        if train_df is None or dev_df is None:
            log.info(f"  [ERROR] Missing data for {l1}, skipping")
            continue

        train_ds, t_mean, t_std = prepare_dataset(train_df, sep_token=sep_token)
        val_ds, _, _ = prepare_dataset(dev_df, sep_token=sep_token, target_mean=t_mean, target_std=t_std)
        scaling_params[l1] = (t_mean, t_std)
        datasets_cache[l1] = (train_ds, val_ds, dev_df)

        log.info(f"  Target scaling: mean={t_mean:.4f}, std={t_std:.4f}")
        log.info(f"  Train: {len(train_ds)} items, Dev: {len(val_ds)} items")

        for seed in SEEDS:
            train_single(l1, seed, train_ds, val_ds, t_mean, t_std, output_dir, tokenizer)

    # ── Generate predictions ──────────────────────────────────────────
    splits_to_predict = ["dev", "test"] if args.predict_split == "both" else [args.predict_split]

    for split in splits_to_predict:
        log.info(f"\n=== PREDICTIONS ({split}) ===\n")

        for l1 in L1S:
            if l1 not in scaling_params:
                continue

            t_mean, t_std = scaling_params[l1]

            # Load split data
            if split == "dev" and l1 in datasets_cache:
                _, eval_ds, eval_df = datasets_cache[l1]
            else:
                eval_df = load_split(data_dir, split, l1)
                if eval_df is None:
                    log.info(f"  [SKIP] No {split} data for {l1}")
                    continue
                eval_ds, _, _ = prepare_dataset(eval_df, sep_token=sep_token, target_mean=t_mean, target_std=t_std)

            item_ids = eval_df["item_id"].values
            all_preds = []

            for seed in SEEDS:
                run_name = f"mdeberta_{l1}_seed{seed}"
                model_dir = output_dir / run_name

                if not model_dir.exists():
                    log.info(f"  [SKIP] {run_name} — no checkpoint found")
                    continue

                log.info(f"  [PREDICT] {run_name} on {split}")
                raw_preds = predict_single(model_dir, eval_ds, tokenizer)

                # Inverse-scale to original GLMM space
                preds = raw_preds * t_std + t_mean
                all_preds.append(preds)

                # Save per-seed predictions
                seed_dir = pred_dir / "per_seed" / split / l1
                seed_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "item_id": item_ids,
                    "prediction": preds,
                }).sort_values("item_id").to_csv(
                    seed_dir / f"mdeberta_seed{seed}_preds.csv", index=False
                )

            if not all_preds:
                log.info(f"  [ERROR] No predictions for {l1} {split}")
                continue

            # Ensemble average
            ensemble_preds = np.mean(all_preds, axis=0)

            # Save ensemble predictions
            ens_dir = pred_dir / "closed" / split / l1
            ens_dir.mkdir(parents=True, exist_ok=True)
            ens_path = ens_dir / "mdeberta_ensemble_preds.csv"
            pd.DataFrame({
                "item_id": item_ids,
                "prediction": ensemble_preds,
            }).sort_values("item_id").to_csv(ens_path, index=False)

            # Evaluate if labels available
            if "GLMM_score" in eval_df.columns:
                y_true = eval_df["GLMM_score"].values
                nan_count = int(np.isnan(ensemble_preds).sum())
                if nan_count > 0:
                    log.info(f"  [WARN] {nan_count} NaN predictions, replacing with mean")
                    mean_pred = np.nanmean(ensemble_preds)
                    ensemble_preds = np.where(np.isnan(ensemble_preds), mean_pred, ensemble_preds)
                    for i in range(len(all_preds)):
                        all_preds[i] = np.where(np.isnan(all_preds[i]), mean_pred, all_preds[i])

                rmse = root_mean_squared_error(y_true, ensemble_preds)
                r, _ = pearsonr(ensemble_preds, y_true)
                log.info(f"\n  {l1.upper()} {split} ensemble: RMSE={rmse:.3f}, Pearson={r:.3f}")

                # Also report per-seed
                for i, seed in enumerate(SEEDS):
                    if i < len(all_preds):
                        s_rmse = root_mean_squared_error(y_true, all_preds[i])
                        s_r, _ = pearsonr(all_preds[i], y_true)
                        log.info(f"    seed {seed}: RMSE={s_rmse:.3f}, Pearson={s_r:.3f}")

            log.info(f"  Saved: {ens_path}")

    # ── Generate report ─────────────────────────────────────────────
    report_lines = [
        "# mDeBERTa Fine-tuning Results",
        "",
        f"**Model:** `{MODEL_NAME}`",
        f"**Seeds:** {SEEDS}",
        f"**Training config:** lr={TRAIN_ARGS['learning_rate']}, epochs={TRAIN_ARGS['num_train_epochs']}, "
        f"batch={TRAIN_ARGS['per_device_train_batch_size']}, weight_decay={TRAIN_ARGS['weight_decay']}, "
        f"scheduler={TRAIN_ARGS['lr_scheduler_type']}, warmup={TRAIN_ARGS['warmup_ratio']}, "
        f"early_stopping=patience 3, target_scaling=yes, max_length={MAX_LENGTH}",
        f"**Input format:** `wlen={{N}} | nedit={{N}} | pos={{POS}} | clue={{N}} | esim={{N}} | L1_word [SEP] L1_context [SEP] clue [SEP] en_word`",
        "",
        "## Results",
        "",
        "| Experiment | L1 | RMSE | Pearson |",
        "|---|---|---|---|",
    ]

    for split in splits_to_predict:
        for l1 in L1S:
            if l1 not in scaling_params:
                continue

            t_mean, t_std = scaling_params[l1]

            # Re-load ensemble predictions
            ens_path = pred_dir / "closed" / split / l1 / "mdeberta_ensemble_preds.csv"
            if not ens_path.exists():
                continue

            ens_df = pd.read_csv(ens_path)

            # Load ground truth
            if split == "dev" and l1 in datasets_cache:
                eval_df = datasets_cache[l1][2]
            else:
                eval_df = load_split(data_dir, split, l1)

            if eval_df is not None and "GLMM_score" in eval_df.columns:
                merged = eval_df[["item_id", "GLMM_score"]].merge(ens_df, on="item_id")
                y_true = merged["GLMM_score"].values
                y_pred = merged["prediction"].values
                rmse = root_mean_squared_error(y_true, y_pred)
                r, _ = pearsonr(y_pred, y_true)
                report_lines.append(f"| mdeberta_ensemble ({split}) | {l1} | {rmse:.3f} | {r:.3f} |")

                # Per-seed rows
                for seed in SEEDS:
                    seed_path = pred_dir / "per_seed" / split / l1 / f"mdeberta_seed{seed}_preds.csv"
                    if seed_path.exists():
                        seed_df = pd.read_csv(seed_path)
                        merged_s = eval_df[["item_id", "GLMM_score"]].merge(seed_df, on="item_id")
                        s_rmse = root_mean_squared_error(merged_s["GLMM_score"].values, merged_s["prediction"].values)
                        s_r, _ = pearsonr(merged_s["prediction"].values, merged_s["GLMM_score"].values)
                        report_lines.append(f"| mdeberta_seed{seed} ({split}) | {l1} | {s_rmse:.3f} | {s_r:.3f} |")

    report_lines += [
        "",
        "## Comparison",
        "",
        "| Model | ES RMSE | DE RMSE | CN RMSE | Avg RMSE |",
        "|---|---|---|---|---|",
        "| XLM-R baseline (closed) | 1.357 | 1.328 | 1.175 | 1.287 |",
        "| full_xgb_v2_embed (feature best) | 1.327 | 1.334 | 1.158 | 1.273 |",
    ]

    # Add ensemble row if available for all L1s
    ens_metrics = {}
    for l1 in L1S:
        ens_path = pred_dir / "closed" / "dev" / l1 / "mdeberta_ensemble_preds.csv"
        if ens_path.exists() and l1 in datasets_cache:
            ens_df = pd.read_csv(ens_path)
            eval_df = datasets_cache[l1][2]
            merged = eval_df[["item_id", "GLMM_score"]].merge(ens_df, on="item_id")
            rmse = root_mean_squared_error(merged["GLMM_score"].values, merged["prediction"].values)
            ens_metrics[l1] = rmse

    if ens_metrics:
        cols = " | ".join(f"**{ens_metrics.get(l1, float('nan')):.3f}**" for l1 in L1S)
        avg = np.mean(list(ens_metrics.values()))
        report_lines.append(f"| **mdeberta_ensemble** | {cols} | **{avg:.3f}** |")

    report_text = "\n".join(report_lines)

    report_path = output_dir / "finetune_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info(f"\nReport saved to {report_path}")
    log.info(f"\n{report_text}")


if __name__ == "__main__":
    main()
