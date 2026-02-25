# AegisML — DS2012 Project Report

## Course: DS2012 — Machine Learning
## Project: Trustworthy Risk + Decision-Support Suite

---

## 1. Title & Team

| Field | Value |
|---|---|
| **Project Title** | AegisML: Trustworthy Risk + Decision-Support Suite |
| **Course Code** | DS2012 |
| **Team Members** | _Fill in_ |
| **Submission Date** | _Fill in_ |

---

## 2. Abstract

AegisML is a full-stack machine-learning platform that integrates supervised, probabilistic,
geometric, neural, evolutionary, fuzzy, reinforcement, and transfer learning models into a
unified decision-support suite. The platform features experiment tracking, model registry,
explainability tools, and a modern web dashboard.

---

## 3. Unit-wise Algorithm Coverage

| DS2012 Unit | Algorithm | Module | Implementation |
|---|---|---|---|
| Unit 1 — Supervised ML | Decision Tree (ID3 + sklearn) | `ml.tabular` | Custom ID3 from scratch + sklearn DT with pruning |
| Unit 2 — Probabilistic | Naive Bayes (GaussianNB, MultinomialNB) | `ml.tabular`, `ml.text` | sklearn |
| Unit 3 — Geometric | SVM (SVC / LinearSVC), kNN (from scratch) | `ml.tabular`, `ml.text` | Custom kNN with `get_neighbors()` for CBR |
| Unit 4 — DT Pruning | `ccp_alpha` cost-complexity pruning | `ml.tabular` | sklearn + rule extraction |
| Unit 5 — Neural Nets | MLP (PyTorch) | `ml.tabular`, `ml.text` | `TabularMLP`, `TextMLP` |
| Unit 6 — Instance-Based | kNN from scratch | `ml.tabular` | Full implementation with distance, voting |
| Unit 7 — Evolutionary | GA feature selection | `ml.ga` | Tournament, crossover, mutation, convergence |
| Unit 8 — Fuzzy Systems | Mamdani fuzzy grading | `ml.fuzzy` | Triangular/trapezoidal MFs, centroid defuzz |
| Unit 9 — RL | Q-learning TicTacToe | `ml.rl` | Tabular Q-learning, epsilon-greedy |
| Unit 10 — Transfer Learning | Text MLP pretrain → finetune | `ml.text` | Pretrain on source domain, finetune on target |

---

## 4. Architecture

![Architecture](figures/architecture.png)

- **Backend**: FastAPI + SQLite experiment tracking
- **Frontend**: Next.js 14 + TypeScript + TailwindCSS
- **ML Engine**: PyTorch + scikit-learn

---

## 5. Dataset Description

| Dataset | Rows | Features | Task | Domain |
|---|---|---|---|---|
| tabular_sample.csv | 2,000 | 11 | Binary (fraud) | Finance |
| text_sample.csv | 1,000 | 2 | Binary (sentiment) | NLP |
| healthcare_synth.csv | 1,500 | 10 | Multi-class (triage) | Healthcare |
| campus_feedback.csv | 300 | 2 | Binary (sentiment) | Education / Transfer |

---

## 6. Results & Evaluation

### 6.1 Tabular Models — Fraud Detection

| Model | Accuracy | F1 | AUC-ROC | Notes |
|---|---|---|---|---|
| Decision Tree | _run experiment_ | | | With `ccp_alpha` pruning |
| Naive Bayes | | | | |
| SVM | | | | |
| kNN (scratch) | | | | k=5 |
| MLP (PyTorch) | | | | 2 hidden layers |

### 6.2 NLP Models — Sentiment

| Model | Accuracy | F1 | Notes |
|---|---|---|---|
| Naive Bayes | | | TF-IDF features |
| SVM | | | |
| MLP | | | |

### 6.3 GA Feature Selection

- **Features selected**: _fill after running_
- **Accuracy (before GA)**: _fill_
- **Accuracy (after GA)**: _fill_
- **Convergence plot**: See `figures/ga_convergence.png`

### 6.4 Fuzzy Grading

Example outputs:

| Attendance | Assignment | Exam | Grade | Score |
|---|---|---|---|---|
| 90 | 85 | 88 | _fill_ | _fill_ |
| 50 | 45 | 40 | _fill_ | _fill_ |

### 6.5 Reinforcement Learning

- **Episodes trained**: _fill_
- **Win rate vs random**: _fill_ %
- **Q-table size**: _fill_ entries

### 6.6 Transfer Learning

| Setting | Accuracy |
|---|---|
| Baseline (train on target only) | _fill_ |
| Pretrain on source + finetune | _fill_ |
| Improvement | _fill_ |

---

## 7. Explainability

### 7.1 Permutation Importance
- Top features for fraud detection: _fill_
- See bar chart in `figures/perm_importance.png`

### 7.2 Decision Tree Rules
- Sample rules extracted: _paste output_

### 7.3 Case-Based Reasoning
- Query: _sample input_
- Similar cases returned with labels and distances

---

## 8. Fairness Analysis

- **Protected attribute**: `gender`
- **Demographic parity gap**: _fill_
- **Interpretation**: _fill_

---

## 9. Screenshots

| Page | Screenshot |
|---|---|
| Home Dashboard | `figures/screenshot_home.png` |
| Model Lab | `figures/screenshot_model_lab.png` |
| Explainability | `figures/screenshot_explainability.png` |
| GA Optimizer | `figures/screenshot_ga.png` |
| Fuzzy Grading | `figures/screenshot_fuzzy.png` |
| RL Arena | `figures/screenshot_rl.png` |
| NLP Studio | `figures/screenshot_nlp.png` |

---

## 10. How to Run

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## 11. Conclusion

_Summarize key findings, which models performed best, insights from explainability and
fairness analysis, and future improvements._

---

## 12. References

1. Scikit-learn documentation — https://scikit-learn.org
2. PyTorch documentation — https://pytorch.org
3. FastAPI documentation — https://fastapi.tiangolo.com
4. Mamdani Fuzzy Inference — E.H. Mamdani, 1975
5. Q-Learning — Watkins & Dayan, 1992
6. DS2012 course materials
