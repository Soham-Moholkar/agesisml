"""AegisML Backend – FastAPI Application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.database import init_db
from app.api import data, training, evaluation, registry, inference, explain, ga, fuzzy, rl, exports, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    await init_db()
    yield


app = FastAPI(
    title="AegisML API",
    description="Trustworthy Risk + Decision Support Suite — DS2012 Machine Learning",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(training.router, prefix="/train", tags=["Training"])
app.include_router(evaluation.router, prefix="/runs", tags=["Evaluation"])
app.include_router(registry.router, prefix="/registry", tags=["Registry"])
app.include_router(inference.router, prefix="/predict", tags=["Inference"])
app.include_router(explain.router, prefix="/explain", tags=["Explainability"])
app.include_router(ga.router, prefix="/ga", tags=["Genetic Algorithm"])
app.include_router(fuzzy.router, prefix="/fuzzy", tags=["Fuzzy"])
app.include_router(rl.router, prefix="/rl", tags=["Reinforcement Learning"])
app.include_router(exports.router, prefix="/export", tags=["Exports"])
