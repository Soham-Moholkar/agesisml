"""Fuzzy grading system – Mamdani-style inference."""
import numpy as np
from typing import Any


# ── Membership Functions ───────────────────────────────────────────

def triangular(x, a, b, c):
    """Triangular membership function."""
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a + 1e-10)
    else:
        return (c - x) / (c - b + 1e-10)


def trapezoidal(x, a, b, c, d):
    """Trapezoidal membership function."""
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a + 1e-10)
    elif b < x <= c:
        return 1.0
    else:
        return (d - x) / (d - c + 1e-10)


# ── Input Fuzzy Sets ──────────────────────────────────────────────

INPUT_SETS = {
    "low":    lambda x: trapezoidal(x, 0, 0, 25, 45),
    "medium": lambda x: triangular(x, 30, 50, 70),
    "high":   lambda x: trapezoidal(x, 55, 75, 100, 100),
}


# ── Output Fuzzy Sets (Grade) ──────────────────────────────────────

GRADE_SETS = {
    "F":  lambda x: trapezoidal(x, 0, 0, 15, 30),
    "D":  lambda x: triangular(x, 20, 35, 50),
    "C":  lambda x: triangular(x, 40, 55, 70),
    "B":  lambda x: triangular(x, 60, 75, 85),
    "A":  lambda x: trapezoidal(x, 80, 90, 100, 100),
}


# ── Fuzzy Rules (Mamdani) ──────────────────────────────────────────

RULES = [
    # (conditions_dict, consequent_grade)
    # Rule 1: If all high → A
    ({"attendance": "high", "assignment": "high", "quiz": "high", "project": "high"}, "A"),
    # Rule 2: If 3 high 1 medium → A
    ({"attendance": "high", "assignment": "high", "quiz": "high", "project": "medium"}, "A"),
    ({"attendance": "high", "assignment": "high", "quiz": "medium", "project": "high"}, "A"),
    ({"attendance": "high", "assignment": "medium", "quiz": "high", "project": "high"}, "A"),
    ({"attendance": "medium", "assignment": "high", "quiz": "high", "project": "high"}, "A"),
    # Rule 3: If 2 high 2 medium → B
    ({"attendance": "high", "assignment": "high", "quiz": "medium", "project": "medium"}, "B"),
    ({"attendance": "high", "assignment": "medium", "quiz": "high", "project": "medium"}, "B"),
    ({"attendance": "high", "assignment": "medium", "quiz": "medium", "project": "high"}, "B"),
    ({"attendance": "medium", "assignment": "high", "quiz": "high", "project": "medium"}, "B"),
    ({"attendance": "medium", "assignment": "high", "quiz": "medium", "project": "high"}, "B"),
    ({"attendance": "medium", "assignment": "medium", "quiz": "high", "project": "high"}, "B"),
    # Rule 4: If mostly medium → C
    ({"attendance": "medium", "assignment": "medium", "quiz": "medium", "project": "medium"}, "C"),
    ({"attendance": "high", "assignment": "medium", "quiz": "medium", "project": "medium"}, "C"),
    ({"attendance": "medium", "assignment": "high", "quiz": "medium", "project": "medium"}, "C"),
    ({"attendance": "medium", "assignment": "medium", "quiz": "high", "project": "medium"}, "C"),
    ({"attendance": "medium", "assignment": "medium", "quiz": "medium", "project": "high"}, "C"),
    # Rule 5: If any low mixed with medium → D
    ({"attendance": "low", "assignment": "medium", "quiz": "medium", "project": "medium"}, "D"),
    ({"attendance": "medium", "assignment": "low", "quiz": "medium", "project": "medium"}, "D"),
    ({"attendance": "medium", "assignment": "medium", "quiz": "low", "project": "medium"}, "D"),
    ({"attendance": "medium", "assignment": "medium", "quiz": "medium", "project": "low"}, "D"),
    ({"attendance": "low", "assignment": "high", "quiz": "medium", "project": "medium"}, "D"),
    ({"attendance": "low", "assignment": "medium", "quiz": "high", "project": "medium"}, "D"),
    ({"attendance": "low", "assignment": "medium", "quiz": "medium", "project": "high"}, "D"),
    # Rule 6: If mostly low → F
    ({"attendance": "low", "assignment": "low", "quiz": "low", "project": "low"}, "F"),
    ({"attendance": "low", "assignment": "low", "quiz": "low", "project": "medium"}, "F"),
    ({"attendance": "low", "assignment": "low", "quiz": "medium", "project": "low"}, "F"),
    ({"attendance": "low", "assignment": "medium", "quiz": "low", "project": "low"}, "F"),
    ({"attendance": "medium", "assignment": "low", "quiz": "low", "project": "low"}, "F"),
    ({"attendance": "low", "assignment": "low", "quiz": "medium", "project": "medium"}, "F"),
    ({"attendance": "low", "assignment": "low", "quiz": "high", "project": "low"}, "F"),
]


def evaluate_fuzzy(attendance: float, assignment: float, quiz: float, project: float):
    """
    Run Mamdani fuzzy inference.
    Returns grade string, numeric score, rule trace, and membership details.
    """
    inputs = {
        "attendance": attendance,
        "assignment": assignment,
        "quiz": quiz,
        "project": project,
    }

    # Fuzzify inputs
    memberships = {}
    for var_name, val in inputs.items():
        memberships[var_name] = {}
        for set_name, mf in INPUT_SETS.items():
            memberships[var_name][set_name] = round(mf(val), 4)

    # Evaluate rules (AND = min, OR = max aggregation)
    grade_activations = {g: 0.0 for g in GRADE_SETS}
    fired_rules = []

    for conditions, grade in RULES:
        # AND: minimum of all condition memberships
        strengths = []
        for var_name, set_name in conditions.items():
            strengths.append(memberships[var_name][set_name])
        firing_strength = min(strengths)
        if firing_strength > 0:
            grade_activations[grade] = max(grade_activations[grade], firing_strength)
            fired_rules.append({
                "conditions": conditions,
                "grade": grade,
                "strength": round(firing_strength, 4),
            })

    # Defuzzify (centroid method)
    x_range = np.linspace(0, 100, 500)
    aggregated = np.zeros_like(x_range)
    for grade, activation in grade_activations.items():
        if activation > 0:
            mf = GRADE_SETS[grade]
            for i, x in enumerate(x_range):
                aggregated[i] = max(aggregated[i], min(activation, mf(x)))

    # Centroid
    if np.sum(aggregated) == 0:
        numeric_score = 50.0  # default
    else:
        numeric_score = float(np.sum(x_range * aggregated) / np.sum(aggregated))

    # Map score to letter grade
    if numeric_score >= 85:
        final_grade = "A"
    elif numeric_score >= 70:
        final_grade = "B"
    elif numeric_score >= 55:
        final_grade = "C"
    elif numeric_score >= 35:
        final_grade = "D"
    else:
        final_grade = "F"

    return {
        "grade": final_grade,
        "numeric_score": round(numeric_score, 2),
        "input_memberships": memberships,
        "grade_activations": {k: round(v, 4) for k, v in grade_activations.items()},
        "fired_rules": fired_rules[:10],  # top fired rules
    }
