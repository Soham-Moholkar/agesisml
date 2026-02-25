"""Application configuration."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "storage" / "datasets"
MODELS_DIR = BASE_DIR / "storage" / "models"
ARTIFACTS_DIR = BASE_DIR / "storage" / "artifacts"
DB_PATH = BASE_DIR / "storage" / "aegisml.db"

SEED = 42

for d in [DATA_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
