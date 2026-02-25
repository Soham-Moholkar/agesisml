"""Inference endpoints — use active models for prediction."""
import numpy as np
import joblib
from fastapi import APIRouter, HTTPException

from app.core.schemas import TabularPredictRequest, TextPredictRequest
from app.core.database import get_active_model, get_run
from app.core.config import MODELS_DIR

router = APIRouter()


@router.post("/tabular")
async def predict_tabular(req: TabularPredictRequest):
    """Predict using active tabular model."""
    active = await get_active_model("tabular")
    if not active:
        raise HTTPException(404, "No active tabular model. Train and activate one first.")

    run_id = active["run_id"]
    run = await get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    pipeline_path = MODELS_DIR / f"{run_id}_pipeline.joblib"
    model_path = MODELS_DIR / f"{run_id}.joblib"

    pipeline = joblib.load(str(pipeline_path))
    feature_names = pipeline["feature_names"]
    model_type = pipeline["model_type"]

    # Build feature vector
    x = np.array([[req.features.get(f, 0) for f in feature_names]], dtype=float)

    # Apply preprocessing
    if pipeline["imputer"]:
        x = pipeline["imputer"].transform(x)
    if pipeline["scaler"]:
        x = pipeline["scaler"].transform(x)

    # Predict
    if model_type == "mlp":
        import torch
        from app.ml.tabular.pipeline import TabularMLP
        config = joblib.load(str(model_path))
        model = TabularMLP(config["input_dim"], n_classes=config["n_classes"])
        model.load_state_dict(torch.load(str(MODELS_DIR / f"{run_id}.pt"), weights_only=True))
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(torch.FloatTensor(x)), dim=1).numpy()
            pred = int(np.argmax(probs, axis=1)[0])
    else:
        model = joblib.load(str(model_path))
        pred = int(model.predict(x)[0])
        probs = model.predict_proba(x).tolist()

    # Decode label
    label = pred
    if pipeline["le_target"]:
        label = pipeline["le_target"].inverse_transform([pred])[0]

    return {
        "prediction": label,
        "prediction_idx": pred,
        "probabilities": probs[0] if isinstance(probs, list) else probs.tolist()[0],
        "model_type": model_type,
        "run_id": run_id,
    }


@router.post("/text")
async def predict_text(req: TextPredictRequest):
    """Predict using active text model."""
    active = await get_active_model("text")
    if not active:
        raise HTTPException(404, "No active text model. Train and activate one first.")

    run_id = active["run_id"]
    text_pipeline_path = MODELS_DIR / f"{run_id}_text_pipeline.joblib"
    model_path = MODELS_DIR / f"{run_id}.joblib"

    text_pipeline = joblib.load(str(text_pipeline_path))
    vectorizer = text_pipeline["vectorizer"]
    le = text_pipeline["le"]
    model_type = text_pipeline["model_type"]

    x = vectorizer.transform([req.text])

    if model_type == "mlp":
        import torch
        from app.ml.text.pipeline import TextMLP
        config = joblib.load(str(model_path))
        model = TextMLP(config["input_dim"], n_classes=config["n_classes"])
        model.load_state_dict(torch.load(str(MODELS_DIR / f"{run_id}.pt"), weights_only=True))
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(torch.FloatTensor(x.toarray())), dim=1).numpy()
            pred = int(np.argmax(probs, axis=1)[0])
    else:
        model = joblib.load(str(model_path))
        pred = int(model.predict(x)[0])
        probs = model.predict_proba(x)

    label = le.inverse_transform([pred])[0]

    return {
        "prediction": label,
        "prediction_idx": pred,
        "probabilities": probs.tolist()[0] if hasattr(probs, 'tolist') else probs[0],
        "model_type": model_type,
        "run_id": run_id,
    }
