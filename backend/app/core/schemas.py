"""Pydantic schemas for request/response models."""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Any


# ── Data ───────────────────────────────────────────────────────────
class DatasetInfo(BaseModel):
    id: str
    filename: str
    n_rows: int
    n_cols: int
    columns: list[str]
    target_column: Optional[str] = None


class SplitRequest(BaseModel):
    target_column: str
    test_size: float = 0.2
    scale: bool = True
    impute: bool = True
    encode_categoricals: bool = True


# ── Training ───────────────────────────────────────────────────────
class TabularTrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset_id: str
    target_column: str
    model_type: str = Field(..., pattern="^(dt|nb|svm|knn|mlp)$")
    test_size: float = 0.2
    scale: bool = True
    impute: bool = True
    encode_categoricals: bool = True
    hyperparams: dict[str, Any] = {}
    seed: int = 42


class TextTrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset_id: str
    text_column: str = "text"
    target_column: str = "label"
    model_type: str = Field(..., pattern="^(nb|svm|mlp)$")
    test_size: float = 0.2
    max_features: int = 5000
    hyperparams: dict[str, Any] = {}
    seed: int = 42


class RLTrainRequest(BaseModel):
    episodes: int = 50000
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.99995
    epsilon_min: float = 0.01
    seed: int = 42


# ── Inference ──────────────────────────────────────────────────────
class TabularPredictRequest(BaseModel):
    task: str = "tabular"
    features: dict[str, Any]


class TextPredictRequest(BaseModel):
    task: str = "text"
    text: str


# ── Explainability ─────────────────────────────────────────────────
class PermutationExplainRequest(BaseModel):
    run_id: str
    n_repeats: int = 10


class CaseExplainRequest(BaseModel):
    run_id: str
    features: dict[str, Any]
    k: int = 5


# ── GA ─────────────────────────────────────────────────────────────
class GARequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset_id: str
    target_column: str
    model_type: str = "dt"
    population_size: int = 30
    generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    seed: int = 42


# ── Fuzzy ──────────────────────────────────────────────────────────
class FuzzyInput(BaseModel):
    attendance: float = Field(..., ge=0, le=100)
    assignment: float = Field(..., ge=0, le=100)
    quiz: float = Field(..., ge=0, le=100)
    project: float = Field(..., ge=0, le=100)


# ── Registry ───────────────────────────────────────────────────────
class ActivateRequest(BaseModel):
    task: str
    run_id: str


# ── RL Play ────────────────────────────────────────────────────────
class RLMoveRequest(BaseModel):
    board: list[int]  # 9 elements: 0=empty, 1=X(human), 2=O(agent)
    run_id: str


# ── Transfer Learning ─────────────────────────────────────────────
class TransferTrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    source_dataset_id: str
    target_dataset_id: str
    text_column: str = "text"
    target_column: str = "label"
    model_type: str = "mlp"
    max_features: int = 5000
    seed: int = 42
