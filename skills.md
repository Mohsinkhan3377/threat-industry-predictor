# skills.md — AI-Assisted Development Workflow

This file documents how Claude (Anthropic) was used as a development assistant throughout this project, and the reusable workflow/instructions that shaped that collaboration.

## Project
Threat Industry Predictor — a multi-class classification model that predicts the most likely targeted industry sector of a cyber threat-intelligence report, deployed as a Flask API (Render) + static web frontend (Netlify).

## How Claude was used (by stage)

### 1. Dataset discovery
- Asked Claude to find current (2025–2026) candidate datasets across several domains (cybersecurity, e-commerce) using web search.
- Evaluated multiple candidates before settling on the AlienVault OTX Threat Intelligence Pulses dataset.
- An initial candidate dataset (a separate IP-reputation dataset) was rejected after EDA showed several key columns were constant across all 10,000 rows — Claude helped identify this and recommended switching datasets rather than forcing a weak target variable.

### 2. Exploratory Data Analysis
- Claude suggested the pandas commands for shape/dtype/missing-value/duplicate inspection and explained how to read the output (e.g. `value_counts()` to check class balance, `isnull().sum()` for missing data).
- Claude flagged which columns were unusable (constant values, near-100%-unknown fields) before any modelling was attempted.

### 3. Data cleaning & encoding decisions
- Each cleaning step was proposed with a stated rationale (e.g. why "Unknown" rows were dropped instead of imputed, why multi-label fields were simplified to a single primary label, why rare categories were grouped into "Other").
- Claude explained the trade-off between one-hot encoding and TF-IDF for different categorical/text fields and why each was chosen for a specific column.

### 4. Model development & evaluation
- Claude wrote the scikit-learn training code for two model families (Logistic Regression, Random Forest) and iterated on feature engineering (adding Title TF-IDF) to improve results.
- Claude helped interpret the classification reports in plain language — explaining what the accuracy/F1 numbers meant relative to a random baseline, and why some classes performed far worse than others (class imbalance / small sample size), rather than just reporting the numbers.

### 5. Deployment engineering
- Claude wrote the Flask backend (`app.py`) that reconstructs the exact training-time feature vector at inference time (scaling, one-hot country encoding, TF-IDF transforms) from a JSON request.
- Claude wrote the static HTML/CSS/JavaScript frontend that calls the API and renders the prediction and top-3 probabilities.
- Claude walked through, step by step, connecting the local project to GitHub, deploying the Flask API on Render, and deploying the static frontend on Netlify.

### 6. Debugging
Claude diagnosed and fixed a series of real deployment issues by asking for the exact terminal output / error text before proposing a fix, rather than guessing:
- A multi-line terminal command that got pasted onto one line and produced a `SyntaxError`.
- Windows File Explorer creating folders instead of files when new files were named through "New Text Document" renaming.
- A Git push rejected due to divergent history, resolved via `git pull --allow-unrelated-histories` and a merge commit.
- A `SyntaxError: invalid syntax` in production caused by a stray Markdown code fence (` ```python `) accidentally left at the top of `app.py` after a copy-paste, found by reading the Render deploy logs traceback line-by-line.
- A front-end `ReferenceError: predict is not defined`, traced to an incomplete file save, fixed by regenerating the file cleanly instead of re-pasting into Notepad.

## Reusable workflow instructions for future ML projects
1. Before modelling, always run `value_counts()` on every candidate target/feature column — a column that looks meaningful in `head()` can still be constant or near-constant across the full dataset.
2. When a text/categorical field is multi-label (comma-separated), decide explicitly whether to simplify to a single label (documented trade-off) or to solve true multi-label classification — don't let `pd.get_dummies` silently explode into hundreds of rare combination columns.
3. Fit all preprocessing objects (scalers, vectorizers, encoders) only on the training split, then serialize every one of them (not just the model) — the inference server needs the exact same transformers, in the exact same feature-column order, as training.
4. When debugging a remote deployment, always fetch the actual server-side traceback/log before proposing a fix; guessing from symptoms alone (e.g. "it's not working") wastes iterations.
5. When copy-pasting AI-generated code into a new file via a plain text editor, verify the first and last lines of the saved file — code fences, truncation, and encoding mismatches are the most common source of otherwise-invisible syntax errors.

## What was *not* delegated to Claude
- Final choice of dataset and target variable was made by the project author after reviewing Claude's findings.
- All Kaggle API credentials, Render/Netlify account setup, and GitHub authentication were performed manually by the author.
- The author reviewed and can explain every preprocessing decision, model hyperparameter, and evaluation metric in this project (see `report.pdf`, Sections 4–10).
