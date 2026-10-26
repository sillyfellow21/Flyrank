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
    "# ML-09 — Hunt leakage; validate honestly\n",
    "\n",
    "**Lane A — Refresh ranking** (FlyRank starter: `content_refresh_anonymized.csv`, 30k rows × 54 cols, 32 clients).\n",
    "Worked in order §0→§6; every number below is computed live in this notebook.\n",
    "\n",
    "> AI use (core first, AI second): AI drafted the leakage/validation code; the leakage taxonomy, split design, attack checklist, and honesty framing are my own, checked against the data.\n",
    "\n",
    "## One-paragraph frame\n",
    "The sneakiest failure in applied ML: the model quietly reads the answer during training. Scores look amazing; the work is worthless. This notebook **attacks its own model before anyone else can** — it demonstrates the three leakage flavors (label-derived, decision-derived/product-flag, and future/overlapping windows), runs the attack checklist, and reports **honest validation**: grouped-by-client and (where possible) time-aware splits, with the base rate printed next to every metric. It also proves the harness works by deliberately adding a leaky feature and watching the score jump toward 1.0, then removing it and keeping the honest number. Claims are framed as **observed / measured / decision-support**, never as predictions of Google's algorithm.\n"
))

cells.append(md(
    "## 0. What this notebook does\n",
    "\n",
    "Per the hunting-leakage-and-validating skill, ML-09 is about **finding label leakage and designing honest validation** before trusting any metric:\n",
    "\n",
    "1. **Leakage taxonomy** — label-derived features, future/overlapping windows, decision-derived product flags\n",
    "2. **Attack checklist** — run it before believing any score\n",
    "3. **Honest splits** — grouped by client (GroupKFold) and report the gap vs a random split\n",
    "4. **Base rate** printed next to every metric\n",
    "5. **Sealed holdout receipts** — the frame-builder script and the metrics file it produced are committed\n",
    "\n",
    "The golden rule: **deliberately add a leaky feature and watch the score jump toward 1.0 — if it doesn't, your test harness is broken.** Then remove it and keep the honest number.\n"
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
    "from sklearn.model_selection import GroupKFold, train_test_split",
    "from sklearn.preprocessing import StandardScaler",
    "from sklearn.pipeline import make_pipeline",
    "",
    "print(\"=\" * 60)",
    "print(\"ML-09 — Hunt leakage; validate honestly\")",
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
    "base_rate = float(df[\"is_declining_label\"].mean())",
    "n_declining = int(df[\"is_declining_label\"].sum())",
    "print(f\"\\ndeclining-label base rate: {base_rate:.4f}  ({n_declining:,} of {len(df):,})\")",
    "print(\"(every metric below is reported next to this base rate)\")",
))