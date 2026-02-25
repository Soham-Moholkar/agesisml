"""Genetic Algorithm endpoints."""
import uuid
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException

from app.core.schemas import GARequest
from app.core.config import DATA_DIR
from app.core.database import get_dataset, insert_run
from app.ml.tabular.pipeline import preprocess_tabular
from app.ml.ga.feature_selection import GeneticFeatureSelector, save_ga_artifacts

router = APIRouter()


@router.post("/feature_select")
async def ga_feature_select(req: GARequest):
    """Run GA feature selection."""
    ds = await get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    filepath = DATA_DIR / f"{req.dataset_id}.csv"
    df = pd.read_csv(str(filepath))

    if req.target_column not in df.columns:
        raise HTTPException(400, f"Target column '{req.target_column}' not found")

    data = preprocess_tabular(
        df, req.target_column, test_size=0.2,
        scale=True, impute=True, encode_categoricals=True, seed=req.seed,
    )

    X_full = np.vstack([data["X_train"], data["X_test"]])
    y_full = np.concatenate([data["y_train"], data["y_test"]])

    ga = GeneticFeatureSelector(
        X_full, y_full, req.model_type,
        req.population_size, req.generations,
        req.crossover_rate, req.mutation_rate,
        seed=req.seed,
    )

    result = ga.run()
    run_id = str(uuid.uuid4())[:8]

    selected_names = save_ga_artifacts(run_id, result, data["feature_names"])

    await insert_run(
        run_id, req.dataset_id, "ga_feature_selection", req.model_type,
        {"population_size": req.population_size, "generations": req.generations},
        {"best_fitness": result["best_fitness"], "n_selected": result["n_selected"],
         "selected_features": selected_names},
        data["feature_names"], req.target_column,
    )

    return {
        "run_id": run_id,
        "selected_features": selected_names,
        "n_selected": result["n_selected"],
        "best_fitness": result["best_fitness"],
        "history": result["history"],
    }
