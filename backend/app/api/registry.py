"""Model registry endpoints."""
from fastapi import APIRouter, HTTPException

from app.core.schemas import ActivateRequest
from app.core.database import activate_model, get_active_model, get_all_active_models, get_run

router = APIRouter()


@router.post("/activate")
async def activate(req: ActivateRequest):
    """Activate a model for a task."""
    run = await get_run(req.run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    await activate_model(req.task, req.run_id)
    return {"task": req.task, "run_id": req.run_id, "status": "activated"}


@router.get("/active")
async def get_active():
    """Get all active models."""
    models = await get_all_active_models()
    return {"active_models": models}


@router.get("/active/{task}")
async def get_active_for_task(task: str):
    """Get active model for a specific task."""
    model = await get_active_model(task)
    if not model:
        raise HTTPException(404, f"No active model for task '{task}'")
    return model
