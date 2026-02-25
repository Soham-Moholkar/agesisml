"""Explainability: permutation importance + case-based reasoning."""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.config import MODELS_DIR, ARTIFACTS_DIR, DATA_DIR
from app.ml.tabular.pipeline import KNNScratch


def compute_permutation_importance(run_id: str, X_test, y_test, feature_names: list,
                                    n_repeats: int = 10, seed: int = 42):
    """Compute permutation importance for a trained model."""
    model_path = MODELS_DIR / f"{run_id}.joblib"
    model = joblib.load(str(model_path))

    # Check if it's an MLP config
    if isinstance(model, dict) and model.get("type") == "mlp":
        # For MLP, use manual permutation importance
        return _manual_permutation_importance(run_id, X_test, y_test, feature_names, n_repeats, seed)

    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                                     random_state=seed, scoring="accuracy")

    importances = result.importances_mean
    idx = np.argsort(importances)[::-1]

    # Save plot
    artifact_dir = ARTIFACTS_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    top_n = min(15, len(feature_names))
    top_idx = idx[:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in reversed(top_idx)],
            importances[list(reversed(top_idx))])
    ax.set_title("Permutation Importance")
    ax.set_xlabel("Mean Accuracy Decrease")
    fig.tight_layout()
    fig.savefig(str(artifact_dir / "permutation_importance.png"), dpi=100)
    plt.close(fig)

    return {
        "importances": {feature_names[i]: float(importances[i]) for i in idx[:top_n]},
        "artifact": f"/runs/{run_id}/artifacts/permutation_importance.png",
    }


def _manual_permutation_importance(run_id, X_test, y_test, feature_names, n_repeats, seed):
    """Manual permutation importance for PyTorch MLP."""
    import torch
    from app.ml.tabular.pipeline import TabularMLP

    config = joblib.load(str(MODELS_DIR / f"{run_id}.joblib"))
    model = TabularMLP(config["input_dim"], n_classes=config["n_classes"])
    model.load_state_dict(torch.load(str(MODELS_DIR / f"{run_id}.pt"), weights_only=True))
    model.eval()

    X_t = torch.FloatTensor(X_test)
    with torch.no_grad():
        base_preds = torch.argmax(model(X_t), dim=1).numpy()
    base_acc = accuracy_score(y_test, base_preds)

    rng = np.random.RandomState(seed)
    importances = np.zeros(X_test.shape[1])

    for feat in range(X_test.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, feat])
            X_p = torch.FloatTensor(X_perm)
            with torch.no_grad():
                preds = torch.argmax(model(X_p), dim=1).numpy()
            scores.append(base_acc - accuracy_score(y_test, preds))
        importances[feat] = np.mean(scores)

    idx = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))
    top_idx = idx[:top_n]

    artifact_dir = ARTIFACTS_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in reversed(top_idx)],
            importances[list(reversed(top_idx))])
    ax.set_title("Permutation Importance (MLP)")
    ax.set_xlabel("Mean Accuracy Decrease")
    fig.tight_layout()
    fig.savefig(str(artifact_dir / "permutation_importance.png"), dpi=100)
    plt.close(fig)

    return {
        "importances": {feature_names[i]: float(importances[i]) for i in top_idx},
        "artifact": f"/runs/{run_id}/artifacts/permutation_importance.png",
    }


def case_based_reasoning(run_id: str, query_features: dict, X_train, y_train,
                          feature_names: list, k: int = 5):
    """Find k nearest neighbors as similar cases for explanation."""
    knn = KNNScratch(k=k)
    knn.fit(X_train, y_train)
    query_vec = [query_features.get(f, 0) for f in feature_names]
    neighbors = knn.get_neighbors(query_vec, k)

    # Enrich with feature names
    for n in neighbors:
        n["feature_values"] = {feature_names[i]: v for i, v in enumerate(n["features"])}
        del n["features"]

    return {
        "query": query_features,
        "similar_cases": neighbors,
        "explanation": f"Found {k} most similar historical cases based on Euclidean distance.",
    }
