"""NLP / Text classification pipeline."""
import numpy as np
import pandas as pd
import joblib
import uuid
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from app.core.config import DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, SEED


# ── Text MLP ───────────────────────────────────────────────────────

class TextMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_text_mlp(X_train, y_train, X_test, input_dim, n_classes=2,
                   hidden_dim=128, lr=0.001, epochs=50, batch_size=64, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TextMLP(input_dim, hidden_dim, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.FloatTensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train)
    y_tr = torch.LongTensor(y_train)
    X_te = torch.FloatTensor(X_test.toarray() if hasattr(X_test, "toarray") else X_test)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_tr.size(0))
        for i in range(0, X_tr.size(0), batch_size):
            idx = perm[i:i + batch_size]
            bx, by = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_te), dim=1).numpy()
        preds = np.argmax(probs, axis=1)
    return model, preds, probs


# ── Pipeline ────────────────────────────────────────────────────────

def preprocess_text(df: pd.DataFrame, text_column: str, target_column: str,
                    max_features: int = 5000, test_size: float = 0.2, seed: int = SEED):
    """TF-IDF + split."""
    texts = df[text_column].fillna("").astype(str).values
    labels = df[target_column].values

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "vectorizer": vectorizer, "le": le,
        "n_classes": n_classes,
        "feature_names": vectorizer.get_feature_names_out().tolist(),
    }


def train_text_model(model_type: str, X_train, y_train, X_test, y_test,
                     n_classes: int, hyperparams: dict, seed: int = SEED):
    """Train a text classifier."""
    np.random.seed(seed)

    if model_type == "nb":
        model = MultinomialNB(alpha=hyperparams.get("alpha", 1.0))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        return model, preds, probs

    elif model_type == "svm":
        base = LinearSVC(C=hyperparams.get("C", 1.0), random_state=seed, max_iter=5000)
        model = CalibratedClassifierCV(base, cv=3)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        return model, preds, probs

    elif model_type == "mlp":
        input_dim = X_train.shape[1]
        model, preds, probs = train_text_mlp(
            X_train, y_train, X_test, input_dim, n_classes,
            hidden_dim=hyperparams.get("hidden_dim", 128),
            lr=hyperparams.get("lr", 0.001),
            epochs=hyperparams.get("epochs", 50),
            seed=seed,
        )
        return model, preds, probs

    raise ValueError(f"Unknown text model type: {model_type}")


def compute_text_metrics(y_true, y_pred, y_prob, n_classes: int):
    """Compute classification metrics for text."""
    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    if n_classes == 2 and y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            pass
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics["confusion_matrix"] = cm
    metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return metrics


def save_text_artifacts(run_id: str, y_true, y_pred, n_classes: int):
    """Save confusion matrix plot for text run."""
    artifact_dir = ARTIFACTS_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Text)")
    fig.tight_layout()
    fig.savefig(str(artifact_dir / "confusion_matrix.png"), dpi=100)
    plt.close(fig)


# ── Transfer Learning ──────────────────────────────────────────────

def transfer_learning_text(source_X_train, source_y_train,
                           target_X_train, target_y_train,
                           target_X_test, target_y_test,
                           input_dim, n_classes=2, seed=42):
    """Simple transfer: pretrain MLP on source, fine-tune on target.
    Shows measurable improvement via faster convergence."""
    torch.manual_seed(seed)

    # Phase 1: Pretrain on source
    model = TextMLP(input_dim, hidden_dim=128, n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    src_X = torch.FloatTensor(source_X_train.toarray() if hasattr(source_X_train, "toarray") else source_X_train)
    src_y = torch.LongTensor(source_y_train)

    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        out = model(src_X)
        loss = criterion(out, src_y)
        loss.backward()
        optimizer.step()

    # Phase 2: Fine-tune on target
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    tgt_X_tr = torch.FloatTensor(target_X_train.toarray() if hasattr(target_X_train, "toarray") else target_X_train)
    tgt_y_tr = torch.LongTensor(target_y_train)
    tgt_X_te = torch.FloatTensor(target_X_test.toarray() if hasattr(target_X_test, "toarray") else target_X_test)

    transfer_history = []
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        out = model(tgt_X_tr)
        loss = criterion(out, tgt_y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(torch.softmax(model(tgt_X_te), dim=1), dim=1).numpy()
            acc = float(accuracy_score(target_y_test, preds))
            transfer_history.append(acc)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tgt_X_te), dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    # Baseline: train from scratch on target only
    baseline_model = TextMLP(input_dim, hidden_dim=128, n_classes=n_classes)
    criterion2 = nn.CrossEntropyLoss()
    optimizer2 = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    baseline_history = []
    for epoch in range(30):
        baseline_model.train()
        optimizer2.zero_grad()
        out = baseline_model(tgt_X_tr)
        loss = criterion2(out, tgt_y_tr)
        loss.backward()
        optimizer2.step()
        baseline_model.eval()
        with torch.no_grad():
            bp = torch.argmax(torch.softmax(baseline_model(tgt_X_te), dim=1), dim=1).numpy()
            baseline_history.append(float(accuracy_score(target_y_test, bp)))

    return {
        "transfer_preds": preds,
        "transfer_probs": probs,
        "transfer_history": transfer_history,
        "baseline_history": baseline_history,
    }
