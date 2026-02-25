"""Tests for kNN from scratch correctness."""
import numpy as np
import pytest
from app.ml.tabular.pipeline import KNNScratch


def test_knn_basic_classification():
    """kNN should classify points correctly on simple separable data."""
    X_train = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],  # class 0
        [5, 5], [5, 6], [6, 5], [6, 6],  # class 1
    ])
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    knn = KNNScratch(k=3)
    knn.fit(X_train, y_train)

    # Points near class 0
    assert knn.predict(np.array([[0.5, 0.5]]))[0] == 0
    # Points near class 1
    assert knn.predict(np.array([[5.5, 5.5]]))[0] == 1


def test_knn_k1():
    """k=1 should return the nearest neighbor's label."""
    X_train = np.array([[0, 0], [10, 10]])
    y_train = np.array([0, 1])
    knn = KNNScratch(k=1)
    knn.fit(X_train, y_train)
    assert knn.predict(np.array([[1, 1]]))[0] == 0
    assert knn.predict(np.array([[9, 9]]))[0] == 1


def test_knn_get_neighbors():
    """get_neighbors should return correct distances and indices."""
    X_train = np.array([[0, 0], [1, 1], [2, 2], [10, 10]])
    y_train = np.array([0, 0, 0, 1])
    knn = KNNScratch(k=2)
    knn.fit(X_train, y_train)

    neighbors = knn.get_neighbors(np.array([0.5, 0.5]), k=2)
    assert len(neighbors) == 2
    assert neighbors[0]["index"] in [0, 1]
    assert neighbors[0]["distance"] < neighbors[1]["distance"] or \
           abs(neighbors[0]["distance"] - neighbors[1]["distance"]) < 1e-6


def test_knn_multiclass():
    """kNN should handle multiclass problems."""
    X = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [5, 6], [10, 0], [11, 0], [10, 1]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    knn = KNNScratch(k=3)
    knn.fit(X, y)
    assert knn.predict(np.array([[0.5, 0.5]]))[0] == 0
    assert knn.predict(np.array([[5.5, 5.5]]))[0] == 1
    assert knn.predict(np.array([[10.5, 0.5]]))[0] == 2
