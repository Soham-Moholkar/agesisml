"""Genetic Algorithm for feature selection."""
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.config import ARTIFACTS_DIR


def get_estimator(model_type: str, seed: int = 42):
    """Get a sklearn estimator for fitness evaluation."""
    if model_type == "dt":
        return DecisionTreeClassifier(max_depth=10, random_state=seed)
    elif model_type == "nb":
        return GaussianNB()
    elif model_type == "svm":
        return SVC(kernel="rbf", random_state=seed)
    elif model_type == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    return DecisionTreeClassifier(max_depth=10, random_state=seed)


class GeneticFeatureSelector:
    """GA for feature selection.

    Chromosome: binary vector of length n_features (1=selected, 0=not).
    Fitness: CV accuracy - penalty * (n_selected / n_total).
    """

    def __init__(self, X, y, model_type="dt",
                 population_size=30, generations=50,
                 crossover_rate=0.8, mutation_rate=0.1,
                 penalty=0.01, seed=42):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_features = self.X.shape[1]
        self.model_type = model_type
        self.pop_size = population_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.penalty = penalty
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _init_population(self):
        """Random binary population. Ensure at least 1 feature per individual."""
        pop = self.rng.randint(0, 2, (self.pop_size, self.n_features))
        for i in range(self.pop_size):
            if pop[i].sum() == 0:
                pop[i, self.rng.randint(self.n_features)] = 1
        return pop

    def _fitness(self, chromosome):
        """Evaluate fitness = CV accuracy - penalty * feature_ratio."""
        selected = np.where(chromosome == 1)[0]
        if len(selected) == 0:
            return 0.0
        X_sel = self.X[:, selected]
        est = get_estimator(self.model_type, self.seed)
        try:
            scores = cross_val_score(est, X_sel, self.y, cv=3, scoring="accuracy")
            cv_score = scores.mean()
        except Exception:
            cv_score = 0.0
        feature_ratio = len(selected) / self.n_features
        return cv_score - self.penalty * feature_ratio

    def _tournament_select(self, pop, fitnesses, k=3):
        """Tournament selection."""
        indices = self.rng.choice(len(pop), k, replace=False)
        best = indices[np.argmax(fitnesses[indices])]
        return pop[best].copy()

    def _crossover(self, p1, p2):
        """Single-point crossover."""
        if self.rng.random() < self.cx_rate:
            pt = self.rng.randint(1, self.n_features)
            c1 = np.concatenate([p1[:pt], p2[pt:]])
            c2 = np.concatenate([p2[:pt], p1[pt:]])
            return c1, c2
        return p1.copy(), p2.copy()

    def _mutate(self, chrom):
        """Bit-flip mutation."""
        for i in range(self.n_features):
            if self.rng.random() < self.mut_rate:
                chrom[i] = 1 - chrom[i]
        if chrom.sum() == 0:
            chrom[self.rng.randint(self.n_features)] = 1
        return chrom

    def run(self):
        """Execute GA. Returns best features, history."""
        pop = self._init_population()
        history = []
        best_ever = None
        best_fitness = -1

        for gen in range(self.generations):
            fitnesses = np.array([self._fitness(ind) for ind in pop])

            gen_best_idx = np.argmax(fitnesses)
            gen_best_fit = fitnesses[gen_best_idx]

            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_ever = pop[gen_best_idx].copy()

            history.append({
                "generation": gen + 1,
                "best_fitness": float(gen_best_fit),
                "avg_fitness": float(fitnesses.mean()),
                "best_n_features": int(pop[gen_best_idx].sum()),
            })

            # Create next generation
            new_pop = [best_ever.copy()]  # elitism
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.extend([c1, c2])
            pop = np.array(new_pop[:self.pop_size])

        selected_indices = np.where(best_ever == 1)[0].tolist()
        return {
            "selected_features": selected_indices,
            "n_selected": len(selected_indices),
            "best_fitness": float(best_fitness),
            "history": history,
        }


def save_ga_artifacts(run_id: str, result: dict, feature_names: list):
    """Save GA convergence plot."""
    artifact_dir = ARTIFACTS_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    history = result["history"]
    gens = [h["generation"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(gens, [h["best_fitness"] for h in history], 'b-', label="Best Fitness")
    ax1.plot(gens, [h["avg_fitness"] for h in history], 'r--', label="Avg Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title("GA Convergence")
    ax1.legend()

    ax2.plot(gens, [h["best_n_features"] for h in history], 'g-')
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("# Features Selected")
    ax2.set_title("Feature Count Over Generations")

    fig.tight_layout()
    fig.savefig(str(artifact_dir / "ga_convergence.png"), dpi=100)
    plt.close(fig)

    # Save selected feature names
    selected = result["selected_features"]
    selected_names = [feature_names[i] for i in selected] if feature_names else selected
    with open(str(artifact_dir / "ga_result.json"), "w") as f:
        json.dump({"selected_features": selected_names, **result}, f, indent=2, default=str)

    return selected_names
