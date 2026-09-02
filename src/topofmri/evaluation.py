from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_classifier(seed=42, kernel="linear", c=1.0, gamma="scale"):
    """Build the SVM; StandardScaler is fitted only on each training split."""
    return make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, kernel=kernel, random_state=seed))


def evaluate(X, y, output_dir, evaluation="holdout", test_size=0.2, folds=5,
             seed=42, kernel="linear", c=1.0, gamma="scale"):
    if evaluation == "cv":
        splits = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y)
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed
        )
        splits = [(train_idx, test_idx)]

    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model = make_classifier(seed + fold, kernel=kernel, c=c, gamma=gamma)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        score = model.decision_function(X[test_idx])
        rows.append({
            "classifier": "svm", "fold": fold,
            "accuracy": accuracy_score(y[test_idx], pred),
            "precision": precision_score(y[test_idx], pred, zero_division=0),
            "recall": recall_score(y[test_idx], pred, zero_division=0),
            "f1": f1_score(y[test_idx], pred, zero_division=0),
            "auc": roc_auc_score(y[test_idx], score),
        })

    per_fold = pd.DataFrame(rows)
    summary = per_fold[["accuracy", "precision", "recall", "f1", "auc"]].agg(["mean", "std"])
    output_dir = Path(output_dir)
    per_fold.to_csv(output_dir / "metrics_per_fold.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv")
    return per_fold, summary
