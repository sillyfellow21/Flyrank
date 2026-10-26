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
    "# ML-10 — Write honest claims\n",
    "\n",
    "**Lane A — Refresh ranking** (FlyRank starter: `content_refresh_anonymized.csv`, 30k rows × 54 cols, 32 clients).\n",
    "Every number below is computed live in this notebook; every claim is written at the rung of the **claim ladder** the evidence can carry.\n",
    "\n",
    "> AI use (core first, AI second): AI drafted the claim-writing scaffolding; the wording, effect-size framing, and honesty checks are my own, checked against the data.\n",
    "\n",
    "## One-paragraph frame\n",
    "Most analysis goes wrong in the sentence, not the math: the words claim more than the numbers showed. This notebook takes the results from ML-07 (rule baseline), ML-08 (model comparison), and ML-09 (leakage hunting) and rewrites each finding in language the evidence can carry — using the **claim ladder** (observed → directional → decision-support, never causal without a design), **effect sizes over drama** (with n's and base rates), and explicit **selection-bias / survivorship / negative-result** checks. Banned phrasings are audited and removed. Claims are framed as **observed / measured / decision-support**, never as predictions of Google's algorithm or causal statements about what \"the algorithm rewards.\"\n"
))

cells.append(md(
    "## 0. The claim ladder — match words to evidence\n",
    "\n",
    "| Evidence you actually have | Words you may use |\n",
    "|---|---|\n",
    "| A pattern in this dataset, this period | \"we observed …\", \"in this data…\" |\n",
    "| A measured comparison between groups | \"X is associated with Y\", \"pages with A showed B\" |\n",
    "| A validated model that ranks/predicts out-of-sample | \"the model ranks/flags … at precision@K of…\" |\n",
    "| A controlled experiment or matched design only | \"X causes/improves Y\" |\n",
    "| Cross-sectional data (one snapshot, no intervention) | NEVER \"doing X will produce Y\" — use decision-support: \"these pages look worth reviewing first, because…\" |\n",
    "\n",
    "Banned unless you ran the design for it: `proves`, `causes`, `will increase`, `the algorithm rewards…`, `we predicted Google's algorithm`.\n",
    "Also banned: any headline ratio from a tiny bucket without reporting `n`; accuracy without its base rate next to it.\n",
))

cells.append(code(
    "import warnings",
    "warnings.filterwarnings(\"ignore\")",
    "",
    "import json",
    "from pathlib import Path",
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
    "print(\"ML-10 — Write honest claims\")",
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
    "# --- label (from trend_direction only) ---",
    "df[\"is_declining_label\"] = (df[\"trend_direction\"].str.lower().eq(\"down\")).astype(int)",
    "BASE_RATE = float(df[\"is_declining_label\"].mean())",
    "N_DECLINING = int(df[\"is_declining_label\"].sum())",
    "N_TOTAL = len(df)",
    "print(f\"\\ndeclining-label base rate: {BASE_RATE:.4f}  ({N_DECLINING:,} of {N_TOTAL:,})\")",
    "print(\"(every metric below is reported next to this base rate)\")",
))