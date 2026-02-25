# AegisML: Trustworthy Risk + Decision Support Suite

> **DS2012: Machine Learning** — End-to-end ML platform demonstrating supervised/unsupervised classification, probabilistic models, geometric models, decision trees, neural networks, evolutionary learning, fuzzy systems, reinforcement learning, and transfer learning.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Problem Statement & Motivation

Financial fraud, fake job postings, and biased lending decisions cost billions annually and erode public trust. **AegisML** is a recruiter-ready ML platform that provides:

- **Fraud / Loan Approval / Fake Job Detection** (tabular classification)
- **NLP Sentiment & Risk Analysis** (text classification)
- **RL TicTacToe Arena** (reinforcement learning showcase)

The platform offers model training, comparison, explainability, GA-driven feature selection, fuzzy grading, and a polished dashboard — all mapped to DS2012 curriculum units.

---

## Unit-Wise Mapping (DS2012)

### Unit 1: Introduction to ML
| Topic | Implementation |
|-------|---------------|
| Predictive vs Descriptive | Model Taxonomy page classifies all models |
| Parametric vs Non-parametric | NB/SVM (parametric) vs kNN/DT (non-parametric) |
| Supervised vs Unsupervised | Classification pipelines (supervised); clustering mention in taxonomy |
| Probabilistic/Geometric/Logical | NB (probabilistic), SVM (geometric), DT (logical) |

### Unit 2: Classification & Neural Networks
| Topic | Implementation |
|-------|---------------|
| Decision Tree (ID3/CART) | Custom ID3 splitter + sklearn DT with pruning |
| Attribute Selection | Information Gain, Gain Ratio, Gini Index |
| Pruning | Cost-complexity pruning (ccp_alpha) |
| Rule Extraction | IF-THEN rules from trained trees |
| Naive Bayes | GaussianNB / MultinomialNB |
| SVM | SVC with kernel selection (linear/rbf/poly) |
| ANN (MLP) | PyTorch MLP for tabular + text |
| Evaluation Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix |

### Unit 3: Advanced Learning Paradigms
| Topic | Implementation |
|-------|---------------|
| Genetic Algorithm | GA feature selection with convergence plots |
| kNN | From-scratch + sklearn comparison |
| Case-Based Reasoning | Nearest neighbor explanation ("similar cases") |
| Fuzzy Systems | Mamdani-style grading system with defuzzification |
| Reinforcement Learning | Q-learning TicTacToe with training curves |
| Transfer Learning | Pretrain sentiment → fine-tune campus feedback |

### Unit 4: Applications
| Topic | Implementation |
|-------|---------------|
| NLP | Sentiment analysis + fake job detection |
| Healthcare | Synthetic triage decision support |
| Big Data Design | Architecture docs for batch/streaming |

---

## Practical Mapping

| Practical | Covered By |
|-----------|-----------|
| P01: Decision Tree Credit Card Fraud | DT module with fraud dataset |
| P02: RL TicTacToe | RL Arena module |
| P03: ANN Stock Prediction | MLP regression demo |
| P04: Twitter Sentiment | NLP text pipeline |
| P05: Fuzzy Grading | Fuzzy module |
| P06: GA Medical Diagnostics | GA feature selection on healthcare data |
| P07-P16: Various | Covered through configurable pipelines |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Backend
```bash
cd aegisml-suite/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd aegisml-suite/frontend
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

### Generate Sample Datasets
```bash
cd aegisml-suite
python scripts/generate_synth_data.py
```

---

## Demo Walkthrough (Viva Script)

1. **Home Page** → Explain AegisML's purpose and DS2012 mapping
2. **Dataset Studio** → Upload `tabular_sample.csv`, preview data, select target
3. **Model Lab** → Train DT, NB, SVM, kNN, MLP; compare metrics
4. **Explainability** → Show feature importance + case-based reasoning
5. **GA Optimizer** → Run GA feature selection, show convergence
6. **Fuzzy Grading** → Adjust sliders, observe grade output
7. **RL Arena** → Show training curve, play against agent
8. **Reports** → Export run summary

---

## Limitations & Future Work

- **Synthetic Data**: Tabular fraud/healthcare datasets are synthetic (clearly marked). Real-world datasets can be swapped in.
- **RNN/LSTM**: MLP regression provided; architecture supports LSTM extension.
- **Computer Vision**: Not implemented; architecture supports addition.
- **Scaling**: SQLite-based; production would use PostgreSQL + model store.
- **Fairness**: Basic demographic parity check; full fairness audit is future work.

---

## License

MIT License — see [LICENSE](LICENSE)
