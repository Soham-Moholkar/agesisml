"""Tests for GA feature selection."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from app.ml.ga.feature_selection import GeneticFeatureSelector


def _make_data(n_informative=5, n_redundant=5, n_features=20, n_samples=200):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=42,
    )
    feature_names = [f"f{i}" for i in range(n_features)]
    return X, y, feature_names


def test_ga_reduces_features():
    """GA should select fewer features than the full set."""
    X, y, names = _make_data()
    ga = GeneticFeatureSelector(
        n_generations=10, pop_size=20, cx_prob=0.7, mut_prob=0.2, random_state=42
    )
    result = ga.run(X, y, names, estimator=DecisionTreeClassifier(random_state=42))
    assert len(result["selected_features"]) < len(names)
    assert len(result["selected_features"]) >= 1


def test_ga_accuracy_reasonable():
    """GA-selected features should yield accuracy >= 0.5 (better than random)."""
    X, y, names = _make_data()
    ga = GeneticFeatureSelector(
        n_generations=15, pop_size=20, cx_prob=0.7, mut_prob=0.2, random_state=42
    )
    result = ga.run(X, y, names, estimator=DecisionTreeClassifier(random_state=42))
    assert result["best_fitness"] >= 0.5


def test_ga_convergence_history():
    """Convergence history should have one entry per generation."""
    X, y, names = _make_data()
    n_gen = 8
    ga = GeneticFeatureSelector(
        n_generations=n_gen, pop_size=10, cx_prob=0.7, mut_prob=0.2, random_state=42
    )
    result = ga.run(X, y, names, estimator=DecisionTreeClassifier(random_state=42))
    assert len(result["convergence"]) == n_gen


def test_ga_deterministic_with_seed():
    """Same seed should give same result."""
    X, y, names = _make_data()
    ga1 = GeneticFeatureSelector(n_generations=5, pop_size=10, random_state=99)
    r1 = ga1.run(X, y, names, estimator=DecisionTreeClassifier(random_state=42))
    ga2 = GeneticFeatureSelector(n_generations=5, pop_size=10, random_state=99)
    r2 = ga2.run(X, y, names, estimator=DecisionTreeClassifier(random_state=42))
    assert r1["selected_features"] == r2["selected_features"]
