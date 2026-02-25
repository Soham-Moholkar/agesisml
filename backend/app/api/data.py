"""Data upload and management endpoints."""
import uuid
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import DATA_DIR
from app.core.database import insert_dataset, get_dataset
from app.core.schemas import SplitRequest

router = APIRouter()


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload CSV dataset and store it."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    ds_id = str(uuid.uuid4())[:8]
    filepath = DATA_DIR / f"{ds_id}.csv"

    content = await file.read()
    with open(str(filepath), "wb") as f:
        f.write(content)

    df = pd.read_csv(str(filepath))
    columns = list(df.columns)

    await insert_dataset(ds_id, file.filename, len(df), len(columns), columns)

    return {
        "dataset_id": ds_id,
        "filename": file.filename,
        "n_rows": len(df),
        "n_cols": len(columns),
        "columns": columns,
    }


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, n_rows: int = 20):
    """Preview first n rows of a dataset."""
    ds = await get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    filepath = DATA_DIR / f"{dataset_id}.csv"
    if not filepath.exists():
        raise HTTPException(404, "Dataset file not found")

    df = pd.read_csv(str(filepath), nrows=n_rows)
    return {
        "dataset_id": dataset_id,
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "n_rows_total": ds["n_rows"],
        "preview": df.fillna("").to_dict(orient="records"),
        "stats": df.describe(include="all").fillna("").to_dict(),
    }


@router.get("/{dataset_id}/columns")
async def get_columns(dataset_id: str):
    """Get column names and types."""
    ds = await get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    filepath = DATA_DIR / f"{dataset_id}.csv"
    df = pd.read_csv(str(filepath), nrows=5)
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
    }


@router.get("/list")
async def list_datasets():
    """List all uploaded datasets."""
    import aiosqlite, json
    from app.core.config import DB_PATH
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM datasets ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [
            {**dict(r), "columns": json.loads(dict(r)["columns"])}
            for r in rows
        ]
