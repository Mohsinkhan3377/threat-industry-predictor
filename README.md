# Threat Industry Predictor

An end-to-end machine learning project that predicts the most likely **targeted industry sector** of a cyber threat-intelligence report (e.g. Government, Finance, Technology, Healthcare) from its title, tags, malware family, and country of origin.

**Live app:** https://spectacular-selkie-0cbeca.netlify.app
**Backend API:** https://threat-industry-predictor.onrender.com

> Note: the backend runs on Render's free tier, which sleeps after ~15 minutes of inactivity. The first prediction after idle time can take 30–60 seconds while it wakes up.

---

## Project Overview

Security teams receive a constant stream of unstructured threat-intelligence reports and must manually work out which industry each one is likely targeting, in order to route it to the right response team. This project trains a multi-class classification model to automate that first triage step, and deploys it as a working web application.

- **Task type:** Supervised Machine Learning — Multi-Class Classification (11 classes)
- **Final model:** Random Forest Classifier (300 trees, max depth 12, class-weight balanced)
- **Final performance:** 43.9% accuracy / 0.412 weighted F1 on the held-out test set (vs. ~9% for random guessing across 11 classes)

## Dataset

- **Source:** AlienVault OTX Threat Intelligence Pulses, via the "Cybersecurity Attacks & Defense Dataset 2026" collection on Kaggle
- **Publisher:** Kaggle user `chuneeb`, sourced from AlienVault OTX
- **URL:** kaggle.com/datasets/chuneeb/ai-cybersecurity-threat-dataset-2026
- **Accessed:** August 9, 2026
- **Size (raw):** 2,365 rows × 14 columns
- **Size (after cleaning):** 1,136 rows used for training/evaluation

Full dataset description, feature dictionary, and licensing notes are in `report.pdf` (Section 2).

## Preprocessing Summary

- Removed rows with an "Unknown" target label (no ground truth to train against)
- Simplified multi-label `Industries` and `Countries` fields to a single primary value, then grouped rare categories into "Other"
- Converted `Malware_Families` into a binary `has_known_malware` flag
- Encoded `country_final` with one-hot encoding
- Encoded `Tags` and `Title` with TF-IDF vectorization (top 30 / top 20 terms)
- Scaled numeric features (`description_length`, `tags_count`) with `StandardScaler`, fit on the training split only
- Addressed class imbalance with `class_weight='balanced'`
- Full before/after cleaning table and rationale for every decision is in `report.pdf` (Sections 3–6)

## Model Development & Evaluation

Two models were trained and compared:

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression | 25.4% | 0.293 |
| Random Forest (Tags only) | 38.6% | 0.382 |
| **Random Forest (Tags + Title) — final** | **43.9%** | **0.412** |

See `report.pdf` (Sections 8–9) for hyperparameters, per-class metrics, and interpretation.

## Architecture

```
Browser (Netlify static site)
      │  fetch() POST /predict  (JSON: title, tags, country, malware flag, description)
      ▼
Flask API (Render, gunicorn)
      │  loads final_model.pkl, tfidf_tags.pkl, tfidf_title.pkl, feature_columns.pkl
      │  reconstructs the exact training-time feature vector
      ▼
Random Forest model.predict() / predict_proba()
      │  returns top prediction + top-3 industries with confidence scores
      ▼
Browser renders result
```

## Repository Structure

```
threat-industry-predictor/
├── BACKEND/              # Flask API + serialized model/preprocessing objects
│   ├── app.py
│   ├── requirements.txt
│   ├── final_model.pkl
│   ├── tfidf_tags.pkl
│   ├── tfidf_title.pkl
│   └── feature_columns.pkl
├── FRONTEND/              # Static web app (deployed to Netlify)
│   └── index.html
├── notebooks/             # Colab notebook: EDA, cleaning, encoding, training, evaluation
│   └── ML_MINI_PROJECT.ipynb
├── skills.md              # AI-assisted workflow documentation
├── README.md
└── report.pdf             # Full project report
```

## How to Run Locally

**Backend:**
```bash
cd BACKEND
pip install -r requirements.txt
python app.py
```
The API will run at `http://localhost:5000`.

**Frontend:**
Open `FRONTEND/index.html` directly in a browser, or serve it with any static file server. Update the `API_URL` constant in `index.html` if pointing at a local backend instead of the deployed Render URL.

**Notebook:**
Open `notebooks/ML_MINI_PROJECT.ipynb` in Google Colab. Requires a Kaggle API token (`kaggle.json`) to download the raw dataset.

## AI-Assisted Development

This project was built with Claude (Anthropic) as a development assistant — for dataset discovery, EDA guidance, cleaning/encoding decisions, model code, the Flask backend, the frontend, and deployment debugging. Full details of what was AI-assisted vs. author-verified are documented in [`skills.md`](./skills.md).

## Author

Mohsin Khan
