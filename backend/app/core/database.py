"""SQLite database for experiment tracking."""
import aiosqlite
import json
from datetime import datetime
from typing import Optional
from app.core.config import DB_PATH

DB_FILE = str(DB_PATH)


async def get_db():
    db = await aiosqlite.connect(DB_FILE)
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                filename TEXT,
                n_rows INTEGER,
                n_cols INTEGER,
                columns TEXT,
                target_column TEXT,
                created_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                dataset_id TEXT,
                task TEXT,
                model_type TEXT,
                hyperparams TEXT,
                metrics TEXT,
                feature_names TEXT,
                target_column TEXT,
                created_at TEXT,
                status TEXT DEFAULT 'completed',
                notes TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS registry (
                task TEXT PRIMARY KEY,
                run_id TEXT,
                activated_at TEXT
            )
        """)
        await db.commit()


async def insert_dataset(ds_id: str, filename: str, n_rows: int, n_cols: int,
                          columns: list, target_column: Optional[str] = None):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT OR REPLACE INTO datasets VALUES (?,?,?,?,?,?,?)",
            (ds_id, filename, n_rows, n_cols, json.dumps(columns),
             target_column, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def get_dataset(ds_id: str):
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM datasets WHERE id=?", (ds_id,))
        row = await cursor.fetchone()
        if row:
            return dict(row)
        return None


async def insert_run(run_id: str, dataset_id: str, task: str, model_type: str,
                      hyperparams: dict, metrics: dict, feature_names: list,
                      target_column: str):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, dataset_id, task, model_type,
             json.dumps(hyperparams), json.dumps(metrics),
             json.dumps(feature_names), target_column,
             datetime.utcnow().isoformat(), "completed", ""),
        )
        await db.commit()


async def get_run(run_id: str):
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM runs WHERE id=?", (run_id,))
        row = await cursor.fetchone()
        if row:
            d = dict(row)
            d["hyperparams"] = json.loads(d["hyperparams"])
            d["metrics"] = json.loads(d["metrics"])
            d["feature_names"] = json.loads(d["feature_names"])
            return d
        return None


async def get_all_runs(task: Optional[str] = None):
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        if task:
            cursor = await db.execute(
                "SELECT * FROM runs WHERE task=? ORDER BY created_at DESC", (task,))
        else:
            cursor = await db.execute("SELECT * FROM runs ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["hyperparams"] = json.loads(d["hyperparams"])
            d["metrics"] = json.loads(d["metrics"])
            d["feature_names"] = json.loads(d["feature_names"])
            results.append(d)
        return results


async def activate_model(task: str, run_id: str):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT OR REPLACE INTO registry VALUES (?,?,?)",
            (task, run_id, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def get_active_model(task: str):
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM registry WHERE task=?", (task,))
        row = await cursor.fetchone()
        if row:
            return dict(row)
        return None


async def get_all_active_models():
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM registry")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
