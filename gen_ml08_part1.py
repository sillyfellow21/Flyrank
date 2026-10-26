import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def md(*lines):
    return {"cell_type": "markdown", "id": "", "metadata": {}, "source": list(lines)}

def code(*lines, exec_count=1, outputs=None):
    return {"cell_type": "code", "execution_count": exec_count, "id": "", "metadata": {}, "outputs": outputs or [], "source": list(lines)}

cells = []

cells.append(md(
    "# ML-08 — Train and compare the model\n",
    "\n",
    "**Lane A — Refresh ranking** (FlyRank starter: `content_refresh_anonymized.csv`, 30k rows × 44 cols, 32 clients).\n",
    "Worked in order §0→§4; every number below is computed live in this notebook.\n",
    "\n",
    "> AI use (core first, AI second): AI drafted the training/evaluation code; the method choice, split design, metric, and honesty checks are my own, checked against the data.\n",
    "\n",
    "## One-paragraph frame\n",
    "For the **content editor** deciding **which pages to refresh first**, we train a first learned model and compare it **honestly** against the ML-07 rule baseline: **same data, same client-holdout split, same metric (precision@K), same base rate.** Because the decision is a ranking (\"which ones first?\"), we score pages with a classifier's **probability of decline** and evaluate with precision@K — not raw accuracy. We then read the errors before believing the score: where the model is wrong, what features it leans on, and three concrete failure cases. Claims are framed as **observed / measured / decision-support**, never as predictions of Google's algorithm.\n"
))

cells.append(md(
    "## 0. Method choice — fit the method to the question\n",
    "\n",
    "Question shape: \"which pages should an editor refresh first?\" — a ranking with an observed label (`trend_direction == \"down\"`).\n",
    "\n",
    "Per the training-honest-models skill, the right choice is:\n",
    "\n",
    "- **Logistic Regression** — readable, fast, strong baseline for yes/no with an observed label\n",
    "- **Random Forest** — readable-ish (feature importances), stronger; its probability feeds the ranking\n",
    "\n",
    "We evaluate both with **precision@K** (share of true decliners in the top-K ranked by predicted probability), the same metric as the ML-07 rule baseline. We do **not** use accuracy — a 54% base rate makes it meaningless.\n",
    "\n",
    "Simplicity is a feature: we start with logistic regression and add Random Forest complexity only if the comparison earns it.\n"
))

cells.append(code(
    "import warnings",
    "warnings.filterwarnings(\"ignore\")",
    "",
    "import numpy as np",
    "import pandas as pd",
    "import sklearn",
    "from sklearn.linear_model import LogisticRegression",
    "from sklearn.ensemble import RandomForestClassifier",
    "from sklearn.model_selection import GroupKFold",
    "from sklearn.preprocessing import StandardScaler",
    "from sklearn.pipeline import make_pipeline",
    "",
    "print(\"=\" * 60)",
    "print(\"ML-08 — Train and compare the model\")",
    "print(\"=\" * 60)",
    "print(f\"scikit-learn version: {sklearn.__version__}\")",
    "print(\"Random seeds fixed: random_state=42\")",
    "",
    "CANDIDATES = [",
    "    \"../../data/raw/content_refresh_anonymized.csv\",  # cwd = work/notebooks",
    "    \"data/raw/content_refresh_anonymized.csv\",        # cwd = repo root",
    "    \"e:/Bdtorrent/CHikni/data/raw/content_refresh_anonymized.csv\",",
    "]",
    "",
    "df = None",
    "for p in CANDIDATES:",
    "    try:",
    "        df = pd.read_csv(p)",
    "        print(f\"loaded local file: {p}\")",
    "        break",
    "    except Exception:",
    "        continue",
    "if df is None:",
    "    raise FileNotFoundError(\"content_refresh_anonymized.csv not found\")",
    "",
    "print(f\"shape (rows, cols): {df.shape}\")",
    "print(f\"clients: {df['client_id'].nunique()}\")",
    "",
    "df[\"is_declining_label\"] = (df[\"trend_direction\"].str.lower().eq(\"down\")).astype(int)",
    "base_rate = float(df[\"is_declining_label\"].mean())",
    "print(f\"\\ndeclining-label base rate: {base_rate:.4f}  ({df['is_declining_label'].sum():,} of {len(df):,})\")",
))