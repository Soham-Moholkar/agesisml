"""Evaluation, runs, and leaderboard endpoints."""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.database import get_run, get_all_runs
from app.core.config import ARTIFACTS_DIR

router = APIRouter()


@router.get("/{run_id}")
async def get_run_details(run_id: str):
    """Get details of a training run."""
    run = await get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@router.get("/{run_id}/artifacts/{filename}")
async def get_artifact(run_id: str, filename: str):
    """Get an artifact (image/file) from a run."""
    artifact_path = ARTIFACTS_DIR / run_id / filename
    if not artifact_path.exists():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(str(artifact_path))


@router.get("/{run_id}/artifacts")
async def list_artifacts(run_id: str):
    """List all artifacts for a run."""
    artifact_dir = ARTIFACTS_DIR / run_id
    if not artifact_dir.exists():
        return {"artifacts": []}
    files = [f.name for f in artifact_dir.iterdir() if f.is_file()]
    return {
        "run_id": run_id,
        "artifacts": files,
        "urls": [f"/runs/{run_id}/artifacts/{f}" for f in files],
    }


@router.get("/")
async def leaderboard(task: str = None):
    """Get leaderboard of all runs, optionally filtered by task."""
    runs = await get_all_runs(task)
    # Sort by accuracy or f1
    runs.sort(key=lambda r: r.get("metrics", {}).get("accuracy", 0), reverse=True)
    return {"runs": runs}
