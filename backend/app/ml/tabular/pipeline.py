"""Tabular ML pipeline: preprocessing, training, evaluation."""
import numpy as np
import pandas as pd
import joblib
import json
import uuid
import os
from pathlib import Path
from typing import Any, Optional

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.config import DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, SEED


# ── Custom ID3 Decision Tree (educational) ─────────────────────────

class ID3Node:
    """Simple ID3 decision tree node for educational purposes."""
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, info_gain=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf class
        self.info_gain = info_gain


def entropy(y):
    """Calculate Shannon entropy."""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))


def gini_index(y):
    """Calculate Gini impurity."""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)


def information_gain(X_col, y, threshold, criterion="entropy"):
    """Calculate information gain for a split."""
    measure = entropy if criterion == "entropy" else gini_index
    parent = measure(y)
    left_mask = X_col <= threshold
    right_mask = ~left_mask
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return 0
    n = len(y)
    child = (left_mask.sum() / n) * measure(y[left_mask]) + \
            (right_mask.sum() / n) * measure(y[right_mask])
    return parent - child


def gain_ratio(X_col, y, threshold, criterion="entropy"):
    """Calculate gain ratio (normalized information gain)."""
    ig = information_gain(X_col, y, threshold, criterion)
    left_mask = X_col <= threshold
    n = len(y)
    p_left = left_mask.sum() / n
    p_right = 1 - p_left
    split_info = -p_left * np.log2(p_left + 1e-10) - p_right * np.log2(p_right + 1e-10)
    return ig / (split_info + 1e-10)


class ID3Tree:
    """Educational ID3/CART-style decision tree built from scratch."""

    def __init__(self, max_depth=5, min_samples=2, criterion="entropy"):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = criterion
        self.root = None
        self.feature_names = None

    def _best_split(self, X, y):
        best_ig = -1
        best_feat = None
        best_thresh = None
        n_features = X.shape[1]
        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                ig = information_gain(X[:, feat_idx], y, thresh, self.criterion)
                if ig > best_ig:
                    best_ig = ig
                    best_feat = feat_idx
                    best_thresh = thresh
        return best_feat, best_thresh, best_ig

    def _build(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))
        if n_classes == 1 or depth >= self.max_depth or n_samples < self.min_samples:
            classes, counts = np.unique(y, return_counts=True)
            return ID3Node(value=classes[np.argmax(counts)])
        feat, thresh, ig = self._best_split(X, y)
        if ig <= 0:
            classes, counts = np.unique(y, return_counts=True)
            return ID3Node(value=classes[np.argmax(counts)])
        left_mask = X[:, feat] <= thresh
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return ID3Node(feature=feat, threshold=thresh, left=left, right=right, info_gain=ig)

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.root = self._build(np.array(X), np.array(y))
        return self

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in np.array(X)])

    def extract_rules(self, node=None, prefix=""):
        """Extract IF-THEN rules from the tree."""
        if node is None:
            node = self.root
        rules = []
        if node.value is not None:
            rules.append(f"{prefix} THEN class = {node.value}")
            return rules
        fname = self.feature_names[node.feature] if self.feature_names else f"feature_{node.feature}"
        left_prefix = f"{prefix} AND {fname} <= {node.threshold:.4f}" if prefix else f"IF {fname} <= {node.threshold:.4f}"
        right_prefix = f"{prefix} AND {fname} > {node.threshold:.4f}" if prefix else f"IF {fname} > {node.threshold:.4f}"
        rules.extend(self.extract_rules(node.left, left_prefix))
        rules.extend(self.extract_rules(node.right, right_prefix))
        return rules


# ── kNN from scratch ───────────────────────────────────────────────

class KNNScratch:
    """k-Nearest Neighbors classifier from scratch."""

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        return self

    def _euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, X):
        X = np.array(X, dtype=float)
        predictions = []
        for x in X:
            distances = [self._euclidean(x, xt) for xt in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            classes, counts = np.unique(k_labels, return_counts=True)
            predictions.append(classes[np.argmax(counts)])
        return np.array(predictions)

    def get_neighbors(self, x, k=None):
        """Return the k nearest neighbors for case-based reasoning."""
        k = k or self.k
        x = np.array(x, dtype=float)
        distances = [self._euclidean(x, xt) for xt in self.X_train]
        k_indices = np.argsort(distances)[:k]
        return [
            {"index": int(idx), "distance": float(distances[idx]),
             "features": self.X_train[idx].tolist(), "label": int(self.y_train[idx])}
            for idx in k_indices
        ]


# ── PyTorch MLP ────────────────────────────────────────────────────

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """Simple MLP for tabular binary/multiclass classification."""

    def __init__(self, input_dim, hidden_dims=(64, 32), n_classes=2, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, y_test, input_dim, n_classes=2,
              hidden_dims=(64, 32), lr=0.001, epochs=100, batch_size=64, seed=42):
    """Train PyTorch MLP and return model + predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TabularMLP(input_dim, hidden_dims, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_te = torch.FloatTensor(X_test)

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_tr.size(0))
        for i in range(0, X_tr.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_tr[indices], y_tr[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_te), dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    return model, preds, probs


# ── Preprocessing pipeline ─────────────────────────────────────────

def preprocess_tabular(df: pd.DataFrame, target_column: str, test_size: float = 0.2,
                        scale: bool = True, impute: bool = True,
                        encode_categoricals: bool = True, seed: int = SEED):
    """Full preprocessing: impute → encode → split → scale."""
    df = df.copy()
    y = df[target_column].values
    X = df.drop(columns=[target_column])

    # Encode target if string
    le_target = None
    if y.dtype == object:
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    feature_names = list(X.columns)

    # Encode categoricals
    label_encoders = {}
    if encode_categoricals:
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    X = X.values.astype(float)

    # Impute
    imputer = None
    if impute:
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Scale
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler, "imputer": imputer,
        "label_encoders": label_encoders,
        "le_target": le_target,
        "n_classes": len(np.unique(y)),
    }


# ── Model training dispatcher ──────────────────────────────────────

def train_model(model_type: str, X_train, y_train, X_test, y_test,
                n_classes: int, feature_names: list, hyperparams: dict, seed: int = SEED):
    """Train a model and return model + predictions + probabilities."""
    np.random.seed(seed)

    if model_type == "dt":
        params = {"max_depth": hyperparams.get("max_depth", 10),
                  "criterion": hyperparams.get("criterion", "gini"),
                  "ccp_alpha": hyperparams.get("ccp_alpha", 0.0),
                  "random_state": seed}
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        return model, preds, probs

    elif model_type == "nb":
        model = GaussianNB()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        return model, preds, probs

    elif model_type == "svm":
        kernel = hyperparams.get("kernel", "rbf")
        C = hyperparams.get("C", 1.0)
        model = SVC(kernel=kernel, C=C, probability=True, random_state=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        return model, preds, probs

    elif model_type == "knn":
        k = hyperparams.get("n_neighbors", 5)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        return model, preds, probs

    elif model_type == "mlp":
        hidden = hyperparams.get("hidden_dims", [64, 32])
        lr = hyperparams.get("lr", 0.001)
        epochs = hyperparams.get("epochs", 100)
        model, preds, probs = train_mlp(
            X_train, y_train, X_test, y_test,
            input_dim=X_train.shape[1], n_classes=n_classes,
            hidden_dims=hidden, lr=lr, epochs=epochs, seed=seed,
        )
        return model, preds, probs

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ── Evaluation + artifacts ──────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, n_classes: int):
    """Compute comprehensive classification metrics."""
    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    # ROC-AUC (binary)
    if n_classes == 2 and y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob[:, 1]))
        except Exception:
            pass
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics["confusion_matrix"] = cm
    # Classification report
    metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return metrics


def save_artifacts(run_id: str, y_true, y_pred, y_prob, n_classes: int, feature_names: list,
                    model, model_type: str):
    """Save confusion matrix plot, ROC curve, and model file."""
    artifact_dir = ARTIFACTS_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(str(artifact_dir / "confusion_matrix.png"), dpi=100)
    plt.close(fig)

    # ROC curve (binary only)
    if n_classes == 2 and y_prob is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, 'b-', label=f"ROC (AUC={roc_auc_score(y_true, y_prob[:, 1]):.3f})")
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("ROC Curve")
            ax.legend()
            fig.tight_layout()
            fig.savefig(str(artifact_dir / "roc_curve.png"), dpi=100)
            plt.close(fig)

            prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob[:, 1])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(rec_vals, prec_vals, 'g-')
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            fig.tight_layout()
            fig.savefig(str(artifact_dir / "pr_curve.png"), dpi=100)
            plt.close(fig)
        except Exception:
            pass

    # Feature importance (for tree-based)
    if model_type == "dt" and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-15:]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([feature_names[i] for i in idx], importances[idx])
        ax.set_title("Feature Importance (DT)")
        fig.tight_layout()
        fig.savefig(str(artifact_dir / "feature_importance.png"), dpi=100)
        plt.close(fig)

    # Save model
    model_path = MODELS_DIR / f"{run_id}.joblib"
    if model_type == "mlp":
        torch.save(model.state_dict(), str(MODELS_DIR / f"{run_id}.pt"))
        # Save model config
        joblib.dump({
            "type": "mlp",
            "input_dim": model.net[0].in_features,
            "n_classes": model.net[-1].out_features,
        }, str(model_path))
    else:
        joblib.dump(model, str(model_path))


def extract_dt_rules(run_id: str):
    """Extract IF-THEN rules from a saved Decision Tree."""
    model_path = MODELS_DIR / f"{run_id}.joblib"
    model = joblib.load(str(model_path))
    if isinstance(model, DecisionTreeClassifier):
        return export_text(model).split("\n")
    return ["Model is not a Decision Tree"]


# ── Fairness check ──────────────────────────────────────────────────

def demographic_parity(y_pred, sensitive_feature):
    """Simple demographic parity check.
    Returns the ratio of positive prediction rates between groups."""
    groups = np.unique(sensitive_feature)
    if len(groups) < 2:
        return {"message": "Need at least 2 groups for fairness check", "parity_ratio": 1.0}
    rates = {}
    for g in groups:
        mask = sensitive_feature == g
        rates[str(g)] = float(np.mean(y_pred[mask]))
    max_rate = max(rates.values())
    min_rate = min(rates.values())
    ratio = min_rate / (max_rate + 1e-10)
    return {
        "group_positive_rates": rates,
        "parity_ratio": float(ratio),
        "passes_80_rule": ratio >= 0.8,
    }
