"""
Absurdly simple baseline: predict GLMM difficulty from word length alone.
Fits a linear regression on train, evaluates on dev.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

DATA_DIR = Path("vocab-difficulty/data")
L1S = ["es", "de", "cn"]

results = []

for l1 in L1S:
    train = pd.read_csv(DATA_DIR / "train" / l1 / f"kvl_shared_task_{l1}_train.csv")
    dev = pd.read_csv(DATA_DIR / "dev" / l1 / f"kvl_shared_task_{l1}_dev.csv")

    # Feature: word length
    X_train = train["en_target_word"].str.len().values.reshape(-1, 1)
    y_train = train["GLMM_score"].values
    X_dev = dev["en_target_word"].str.len().values.reshape(-1, 1)
    y_dev = dev["GLMM_score"].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_dev)

    rmse = root_mean_squared_error(y_dev, preds)
    r, _ = pearsonr(preds, y_dev)

    coef, intercept = model.coef_[0], model.intercept_
    results.append({"L1": l1, "RMSE": rmse, "Pearson": r,
                     "coef": coef, "intercept": intercept})

    print(f"[{l1.upper()}] RMSE={rmse:.3f}  Pearson={r:.3f}  "
          f"(score = {coef:.3f} * word_len + {intercept:.3f})")

print("\n--- Comparison with XLM-R baseline (closed track) ---")
baseline = {"es": 1.357, "de": 1.328, "cn": 1.175}
for r in results:
    bl = baseline[r["L1"]]
    diff = r["RMSE"] - bl
    print(f"  {r['L1'].upper()}: word_length={r['RMSE']:.3f}  baseline={bl:.3f}  "
          f"gap={diff:+.3f} ({diff/bl*100:+.1f}%)")
