"""Training endpoints for tabular, text, and RL."""
import uuid
import pandas as pd
import numpy as np
import joblib
import json
from fastapi import APIRouter, HTTPException

from app.core.config import DATA_DIR, MODELS_DIR, ARTIFACTS_DIR
from app.core.schemas import TabularTrainRequest, TextTrainRequest, RLTrainRequest, TransferTrainRequest
from app.core.database import insert_run, get_dataset

from app.ml.tabular.pipeline import (
    preprocess_tabular, train_model, compute_metrics,
    save_artifacts, extract_dt_rules,
)
from app.ml.text.pipeline import (
    preprocess_text, train_text_model, compute_text_metrics,
    save_text_artifacts, transfer_learning_text,
)
from app.ml.rl.tictactoe import train_rl_agent, save_rl_agent

router = APIRouter()


@router.post("/tabular")
async def train_tabular(req: TabularTrainRequest):
    """Train a tabular classifier."""
    ds = await get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    filepath = DATA_DIR / f"{req.dataset_id}.csv"
    df = pd.read_csv(str(filepath))

    if req.target_column not in df.columns:
        raise HTTPException(400, f"Target column '{req.target_column}' not found")

    # Preprocess
    data = preprocess_tabular(
        df, req.target_column, req.test_size,
        req.scale, req.impute, req.encode_categoricals, req.seed,
    )

    # Train
    model, preds, probs = train_model(
        req.model_type, data["X_train"], data["y_train"],
        data["X_test"], data["y_test"], data["n_classes"],
        data["feature_names"], req.hyperparams, req.seed,
    )

    # Metrics
    metrics = compute_metrics(data["y_test"], preds, probs, data["n_classes"])

    # Cross-validation score
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    est_map = {
        "dt": lambda: DecisionTreeClassifier(
            max_depth=req.hyperparams.get("max_depth", 10),
            random_state=req.seed),
        "nb": lambda: GaussianNB(),
        "svm": lambda: SVC(
            kernel=req.hyperparams.get("kernel", "rbf"),
            random_state=req.seed),
        "knn": lambda: KNeighborsClassifier(
            n_neighbors=req.hyperparams.get("n_neighbors", 5)),
    }
    if req.model_type in est_map:
        X_full = np.vstack([data["X_train"], data["X_test"]])
        y_full = np.concatenate([data["y_train"], data["y_test"]])
        cv_scores = cross_val_score(est_map[req.model_type](), X_full, y_full, cv=5)
        metrics["cv_accuracy_mean"] = float(cv_scores.mean())
        metrics["cv_accuracy_std"] = float(cv_scores.std())

    # Save
    run_id = str(uuid.uuid4())[:8]
    save_artifacts(run_id, data["y_test"], preds, probs, data["n_classes"],
                    data["feature_names"], model, req.model_type)

    # Save preprocessing pipeline
    joblib.dump({
        "scaler": data["scaler"],
        "imputer": data["imputer"],
        "label_encoders": data["label_encoders"],
        "le_target": data["le_target"],
        "feature_names": data["feature_names"],
        "n_classes": data["n_classes"],
        "model_type": req.model_type,
    }, str(MODELS_DIR / f"{run_id}_pipeline.joblib"))

    # Save train/test data for explainability
    joblib.dump({
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
    }, str(MODELS_DIR / f"{run_id}_data.joblib"))

    # Extract rules for DT
    rules = []
    if req.model_type == "dt":
        rules = extract_dt_rules(run_id)

    await insert_run(
        run_id, req.dataset_id, "tabular", req.model_type,
        req.hyperparams, metrics, data["feature_names"], req.target_column,
    )

    return {
        "run_id": run_id,
        "model_type": req.model_type,
        "metrics": metrics,
        "rules": rules[:50] if rules else [],
    }


@router.post("/text")
async def train_text(req: TextTrainRequest):
    """Train a text classifier."""
    ds = await get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    filepath = DATA_DIR / f"{req.dataset_id}.csv"
    df = pd.read_csv(str(filepath))

    if req.text_column not in df.columns or req.target_column not in df.columns:
        raise HTTPException(400, "Required columns not found")

    data = preprocess_text(df, req.text_column, req.target_column,
                            req.max_features, req.test_size, req.seed)

    model, preds, probs = train_text_model(
        req.model_type, data["X_train"], data["y_train"],
        data["X_test"], data["y_test"], data["n_classes"],
        req.hyperparams, req.seed,
    )

    metrics = compute_text_metrics(data["y_test"], preds, probs, data["n_classes"])

    run_id = str(uuid.uuid4())[:8]
    save_text_artifacts(run_id, data["y_test"], preds, data["n_classes"])

    # Save model + vectorizer
    if req.model_type == "mlp":
        import torch
        torch.save(model.state_dict(), str(MODELS_DIR / f"{run_id}.pt"))
        joblib.dump({
            "type": "text_mlp",
            "input_dim": data["X_train"].shape[1],
            "n_classes": data["n_classes"],
        }, str(MODELS_DIR / f"{run_id}.joblib"))
    else:
        joblib.dump(model, str(MODELS_DIR / f"{run_id}.joblib"))

    joblib.dump({
        "vectorizer": data["vectorizer"],
        "le": data["le"],
        "model_type": req.model_type,
        "n_classes": data["n_classes"],
    }, str(MODELS_DIR / f"{run_id}_text_pipeline.joblib"))

    await insert_run(
        run_id, req.dataset_id, "text", req.model_type,
        req.hyperparams, metrics, data["feature_names"][:20], req.target_column,
    )

    return {
        "run_id": run_id,
        "model_type": req.model_type,
        "metrics": metrics,
    }


@router.post("/rl/tictactoe")
async def train_rl(req: RLTrainRequest):
    """Train RL agent for TicTacToe."""
    agent, stats, history = train_rl_agent(
        req.episodes, req.alpha, req.gamma,
        req.epsilon, req.epsilon_decay, req.epsilon_min, req.seed,
    )

    run_id = str(uuid.uuid4())[:8]
    save_rl_agent(run_id, agent, stats, history)

    await insert_run(
        run_id, "", "rl_tictactoe", "q_learning",
        {"episodes": req.episodes, "alpha": req.alpha, "gamma": req.gamma},
        {"wins": stats["wins"], "losses": stats["losses"], "draws": stats["draws"],
         "win_rate": stats["wins"] / req.episodes,
         "q_table_size": len(agent.q_table)},
        [], "",
    )

    return {
        "run_id": run_id,
        "stats": stats,
        "history": history[-10:],
        "q_table_size": len(agent.q_table),
    }


@router.post("/transfer")
async def train_transfer(req: TransferTrainRequest):
    """Transfer learning: pretrain on source, fine-tune on target."""
    src_ds = await get_dataset(req.source_dataset_id)
    tgt_ds = await get_dataset(req.target_dataset_id)
    if not src_ds or not tgt_ds:
        raise HTTPException(404, "Dataset not found")

    src_df = pd.read_csv(str(DATA_DIR / f"{req.source_dataset_id}.csv"))
    tgt_df = pd.read_csv(str(DATA_DIR / f"{req.target_dataset_id}.csv"))

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Build shared vocabulary
    all_text = pd.concat([
        src_df[req.text_column].fillna(""),
        tgt_df[req.text_column].fillna(""),
    ])
    vectorizer = TfidfVectorizer(max_features=req.max_features, stop_words="english")
    vectorizer.fit(all_text)

    src_X = vectorizer.transform(src_df[req.text_column].fillna(""))
    le_src = LabelEncoder()
    src_y = le_src.fit_transform(src_df[req.target_column])

    tgt_X = vectorizer.transform(tgt_df[req.text_column].fillna(""))
    le_tgt = LabelEncoder()
    tgt_y = le_tgt.fit_transform(tgt_df[req.target_column])

    tgt_X_train, tgt_X_test, tgt_y_train, tgt_y_test = train_test_split(
        tgt_X, tgt_y, test_size=0.3, random_state=req.seed, stratify=tgt_y,
    )

    n_classes = len(np.unique(tgt_y))
    result = transfer_learning_text(
        src_X, src_y, tgt_X_train, tgt_y_train, tgt_X_test, tgt_y_test,
        input_dim=req.max_features, n_classes=n_classes, seed=req.seed,
    )

    from sklearn.metrics import accuracy_score
    transfer_acc = float(accuracy_score(tgt_y_test, result["transfer_preds"]))

    run_id = str(uuid.uuid4())[:8]

    # Save convergence plot
    artifact_dir = ARTIFACTS_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result["transfer_history"], label="With Transfer", color="blue")
    ax.plot(result["baseline_history"], label="From Scratch", color="red", linestyle="--")
    ax.set_xlabel("Fine-tuning Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Transfer Learning vs From Scratch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(artifact_dir / "transfer_learning.png"), dpi=100)
    plt.close(fig)

    metrics = {
        "transfer_final_accuracy": transfer_acc,
        "baseline_final_accuracy": result["baseline_history"][-1] if result["baseline_history"] else 0,
        "transfer_history": result["transfer_history"],
        "baseline_history": result["baseline_history"],
    }

    await insert_run(
        run_id, req.target_dataset_id, "transfer", "mlp_transfer",
        {"source_dataset": req.source_dataset_id}, metrics, [], req.target_column,
    )

    return {"run_id": run_id, "metrics": metrics}
