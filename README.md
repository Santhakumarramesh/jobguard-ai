# 🛡️ JobGuard AI — Super Portal

**Free & public** fake job posting detector. Paste any job listing — get instant REAL or FAKE verdict with fraud probability. No signup required.

| Feature | Description |
|---------|-------------|
| 🔍 Live analyzer | Paste job text → get verdict |
| 📊 Model report | Accuracy, ROC-AUC, confusion matrix |
| 📖 How it works | 3-step pipeline explained |
| ❓ FAQ | Common questions |
| 🚩 Red flags | Scam warning signs guide |
| 📋 Copy / Share | Export results |

## 🔬 Model Details

| Metric | Value |
|--------|-------|
| Model | Random Forest (200 trees) |
| Accuracy | **98.4%** |
| ROC-AUC | **0.988** |
| F1 Score | **0.837** (threshold optimized) |
| Precision | 97.5% |
| Recall | 74.0% |
| CV F1 | 0.666 ± 0.048 (5-fold) |

**Training Dataset:** [Kaggle Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-predicting)
- 17,880 job postings
- 4.84% fraud rate
- Class balancing applied

## 🧠 Real Model Predictions (from notebook)

Portal displays actual outputs from `jobguard-classifier.ipynb`:

| Job | Verdict | Fraud Prob |
|-----|---------|------------|
| Senior Software Engineer (Python, AWS) | ✅ LEGITIMATE | 39.6% |
| EARN $10,000 WEEKLY, no skills needed | 🚨 FRAUDULENT | 52.8% |
| Marketing Manager, B2B SaaS | ✅ LEGITIMATE | 38.7% |
| Data Entry Clerk $500/day guaranteed | 🚨 FRAUDULENT | 64.8% |
| UX Designer, Figma expertise | ✅ LEGITIMATE | 30.4% |

## ⚙️ ML Pipeline

1. **Load data** — Primary: 17,880 rows. For more accuracy, run:
   ```bash
   python3 scripts/download_datasets.py   # Merge Kaggle + HuggingFace
   python3 scripts/augment_fraud_data.py # Augment fraud samples → ~18,700 rows, 9% fraud
   ```
2. **Text Cleaning** — NLTK, lemmatization, stopwords removal
3. **Feature Engineering** — TF-IDF 5,000 features + 9 meta-features
4. **Class Balancing** — 50/50 fraud/legit training split with oversampling
5. **5-Fold CV Training** — Random Forest selected as best
6. **Threshold Optimization** — 0.50 → 0.47 (F1: 0.803 → 0.837)

## 🚀 Live Demo & Deployment

**Live at:** [https://santhakumarramesh.github.io/jobguard-ai/](https://santhakumarramesh.github.io/jobguard-ai/)

### Create Repo & Deploy

```bash
# 1. Initialize (if needed) and commit
git add .
git commit -m "Deploy JobGuard AI"

# 2. Create repo on GitHub: github.com/new → name: jobguard-ai

# 3. Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/jobguard-ai.git
git branch -M main
git push -u origin main

# 4. Enable Pages: Repo → Settings → Pages → Source: GitHub Actions
# 5. Site goes live at https://YOUR_USERNAME.github.io/jobguard-ai
```

The portal includes a **live analyzer** that outputs REAL or FAKE verdicts.

## 🧩 Reusable Python Package

The notebook now delegates the core logic to the `jobguard/` package:

- `jobguard/text.py` for text cleaning and feature extraction
- `jobguard/pipeline.py` for training, evaluation, threshold selection, and artifact export
- `jobguard/detector.py` for loading trained artifacts and scoring job posts
- `jobguard/heuristics.py` for the rule-based fallback detector
- `jobguard/api.py` for the FastAPI inference service

The notebook remains the report and experiment runner, while the reusable code is importable from Python.

## 🔌 Inference API

Run the API locally with:

```bash
uvicorn jobguard.api:app --reload
```

Key endpoints:

- `GET /`
- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict/batch`

The API accepts either raw job text or structured job fields and falls back to the rule-based detector if trained artifacts are unavailable.

## ✅ CI Smoke Tests

GitHub Actions runs the smoke suite on every push and pull request:

- notebook validation and compilation
- deployment workflow validation
- API request/response checks
- pipeline training and threshold smoke test

## 📁 Project Structure

```
jobguard-ai/
├── index.html              ← Portal (live analyzer, model output)
├── jobguard-dataset.csv    ← Primary Kaggle dataset (17,880 rows)
├── jobguard-classifier.ipynb
├── jobguard/               ← Reusable package
├── tests/                  ← Smoke tests for notebook, API, pipeline, and workflow
├── scripts/
│   ├── download_datasets.py
│   └── augment_fraud_data.py
├── requirements.txt
├── data/                   ← Generated (jobguard-augmented.csv, etc.)
├── DEPLOY.md
├── ENABLE_PAGES.md
├── ADD_WORKFLOW.md
└── .github/workflows/
    ├── ci.yml
    └── deploy.yml
```

## 🔑 Top Features (Random Forest Importance)

1. `has_company_logo` — **#1 predictor** (fraud rate: 15.9% without, 2.0% with)
2. `word_count` — Fraud posts average 282 words vs legit 379
3. `text_length` — Correlated with word_count
4. TF-IDF: `growing`, `web`, `data entry`, `work home`, `earn`, `encouraged`
5. `has_questions` — Fraud rate 6.8% without vs 2.8% with

## 📊 Confusion Matrix

```
Actual\Predicted  | Legitimate | Fraudulent
------------------|------------|------------
Legitimate        |   3,400 ✅ |       3 ❌
Fraudulent        |      55 ❌ |     118 ✅
```

## 🛠️ Technologies

- **Frontend:** Pure HTML5, CSS3, Vanilla JS (single file, no dependencies)
- **Detector:** Reusable Python inference package with a heuristic fallback and FastAPI service
- **Model Output:** Verdict (REAL/FAKE/SUSPICIOUS), fraud probability, risk level, signals
- **Deployment:** GitHub Pages (auto-deploy on push to main)

---

*Built from Kaggle dataset · Model: jobguard-classifier.ipynb*
