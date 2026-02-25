"""Tests for decision tree rule extraction."""
import pytest
import numpy as np
from sklearn.datasets import make_classification
from app.ml.tabular.pipeline import preprocess_tabular, train_model, extract_dt_rules


def _train_dt():
    """Train a simple DT and return artifacts."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    import pandas as pd
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["target"] = y
    artifacts = train_model(
        df, target_col="target", algorithm="decision_tree",
        test_size=0.2, random_state=42
    )
    return artifacts


def test_rules_are_list_of_strings():
    arts = _train_dt()
    rules = extract_dt_rules(arts["model"], arts["feature_names"])
    assert isinstance(rules, list)
    assert all(isinstance(r, str) for r in rules)


def test_rules_contain_feature_names():
    arts = _train_dt()
    rules = extract_dt_rules(arts["model"], arts["feature_names"])
    combined = " ".join(rules)
    # At least one feature should appear
    found = any(f in combined for f in arts["feature_names"])
    assert found, "No feature names found in rule text"


def test_rules_contain_class_keyword():
    arts = _train_dt()
    rules = extract_dt_rules(arts["model"], arts["feature_names"])
    combined = " ".join(rules)
    assert "class:" in combined.lower() or "Class:" in combined


def test_rules_nonempty():
    arts = _train_dt()
    rules = extract_dt_rules(arts["model"], arts["feature_names"])
    assert len(rules) >= 1
