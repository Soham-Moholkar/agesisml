"""Tests for fuzzy grading system."""
import pytest
from app.ml.fuzzy.grading import fuzzy_grade


def test_high_scores_give_A():
    result = fuzzy_grade(attendance=95, assignment=92, exam=90)
    assert result["grade"] in ("A", "B"), f"Got {result['grade']} for high scores"


def test_low_scores_give_F():
    result = fuzzy_grade(attendance=10, assignment=15, exam=12)
    assert result["grade"] in ("F", "D"), f"Got {result['grade']} for low scores"


def test_medium_scores_give_C():
    result = fuzzy_grade(attendance=50, assignment=50, exam=50)
    assert result["grade"] in ("C", "B", "D"), f"Got {result['grade']} for medium scores"


def test_score_in_range():
    result = fuzzy_grade(attendance=70, assignment=60, exam=65)
    assert 0 <= result["score"] <= 100


def test_fired_rules_present():
    result = fuzzy_grade(attendance=80, assignment=75, exam=70)
    assert "fired_rules" in result
    assert len(result["fired_rules"]) >= 1


def test_memberships_present():
    result = fuzzy_grade(attendance=60, assignment=60, exam=60)
    assert "memberships" in result
    assert "attendance" in result["memberships"]
