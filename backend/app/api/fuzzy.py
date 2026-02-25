"""Fuzzy grading system endpoint."""
from fastapi import APIRouter

from app.core.schemas import FuzzyInput
from app.ml.fuzzy.grading import evaluate_fuzzy

router = APIRouter()


@router.post("/grade")
async def fuzzy_grade(inp: FuzzyInput):
    """Evaluate fuzzy grading system."""
    result = evaluate_fuzzy(inp.attendance, inp.assignment, inp.quiz, inp.project)
    return result
