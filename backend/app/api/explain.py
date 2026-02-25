"""Explainability endpoints."""
import numpy as np
import joblib
from fastapi import APIRouter, HTTPException

from app.core.schemas import PermutationExplainRequest, CaseExplainRequest
from app.core.config import MODELS_DIR
from app.core.database import get_run
from app.ml.explain.explainability import compute_permutation_importance, case_based_reasoning
from app.ml.tabular.pipeline import extract_dt_rules

router = APIRouter()


@router.post("/tabular/permutation")
async def permutation_explain(req: PermutationExplainRequest):
    """Compute permutation importance for a trained model."""
    run = await get_run(req.run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    data_path = MODELS_DIR / f"{req.run_id}_data.joblib"
    if not data_path.exists():
        raise HTTPException(404, "Training data not available for this run")

    data = joblib.load(str(data_path))
    pipeline = joblib.load(str(MODELS_DIR / f"{req.run_id}_pipeline.joblib"))

    result = compute_permutation_importance(
        req.run_id, data["X_test"], data["y_test"],
        pipeline["feature_names"], req.n_repeats,
    )

    return result


@router.post("/tabular/cases")
async def case_explain(req: CaseExplainRequest):
    """Find similar cases using kNN for case-based reasoning."""
    run = await get_run(req.run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    data_path = MODELS_DIR / f"{req.run_id}_data.joblib"
    if not data_path.exists():
        raise HTTPException(404, "Training data not available for this run")

    data = joblib.load(str(data_path))
    pipeline = joblib.load(str(MODELS_DIR / f"{req.run_id}_pipeline.joblib"))

    result = case_based_reasoning(
        req.run_id, req.features, data["X_train"], data["y_train"],
        pipeline["feature_names"], req.k,
    )

    return result


@router.get("/tree/rules/{run_id}")
async def tree_rules(run_id: str):
    """Extract IF-THEN rules from a decision tree."""
    run = await get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    if run["model_type"] != "dt":
        raise HTTPException(400, "Rules only available for Decision Tree models")

    rules = extract_dt_rules(run_id)
    return {"run_id": run_id, "rules": rules}
