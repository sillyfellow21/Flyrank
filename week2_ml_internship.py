#!/usr/bin/env python3
"""
Week 2 — The model is just a rule you can read
FlyRank ML Internship (Applied Search Intelligence track)

Task list:
  1. Setup: Load data, define is_declining_label
  2. Task 1: Write a 1-line hand rule "stale x visible" and rank pages
  3. Task 2: Fit a depth-2 decision tree, print it, compare with the rule
  4. Task 3: Leakage check — never feed the outcome back in as a feature
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/raw/content_refresh_anonymized.csv")
OUTPUT_PATH = Path("week2_results.md")
RESULTS_JSON_PATH = Path("week2_results.json")

LEAKAGE_COLUMNS = ["trend_direction", "trend_pct"]

MODEL_NUMERIC_FEATURES = [
    "impressions_90d",
    "clicks_90d",
    "sessions_90d",
    "ai_sessions_90d",
    "days_since_last_update",
    "content_age_days",
    "ctr",
    "avg_position",
    "engagement_rate",
    "scroll_rate",
]

# Hand-rule parameters
STALE_DAYS = 90
MIN_IMPRESSIONS = 100

print("=" * 60)
print("WEEK 2 — The model is just a rule you can read")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_PATH}\n"
        "Expected: data/raw/content_refresh_anonymized.csv"
    )

df = pd.read_csv(DATA_PATH)

# Define the label
df["is_declining_label"] = (
    df["trend_direction"].str.lower().eq("down").astype(int)
)

n_pages = df.shape[0]
decline_rate = df["is_declining_label"].mean()
print(f"Loaded {n_pages:,} pages | declining rate: {decline_rate:.3f}")

# ---------------------------------------------------------------------------
# Task 1 — Hand rule: "stale x visible"
# ---------------------------------------------------------------------------
print("\n[TASK 1] Hand rule: stale x visible")
print("  Intuition: stale (days_since_last_update > STALE_DAYS) AND")
print("            visible (impressions_90d >= MIN_IMPRESSIONS),")
print("            ranked by impressions_90d (exposure).")

df["hand_rule_score"] = np.where(
    (df["days_since_last_update"] > STALE_DAYS)
    & (df["impressions_90d"] >= MIN_IMPRESSIONS),
    df["impressions_90d"],
    0,
)

df_sorted = df.sort_values("hand_rule_score", ascending=False).reset_index(drop=True)

print(f"  STALE_DAYS={STALE_DAYS}, MIN_IMPRESSIONS={MIN_IMPRESSIONS}")
print(f"  Pages flagged: {(df['hand_rule_score'] > 0).sum():,}")
print("  Top 5 by hand-rule score:")

# ---------------------------------------------------------------------------
# Task 2 — Depth-2 decision tree
# ---------------------------------------------------------------------------
print("\n[TASK 2] Depth-2 decision tree")
print("  Fit a simple decision tree and print the if/else rules.")

X = df[MODEL_NUMERIC_FEATURES].copy()
y = df["is_declining_label"]

# Grouped split by client to avoid data leakage across clients
client_train, client_test = train_test_split(
    df["client_id"].unique(),
    test_size=0.2,
    random_state=42,
)

train_mask = df["client_id"].isin(client_train)
test_mask = df["client_id"].isin(client_test)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"  Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

# Depth-2 decision tree
tree = DecisionTreeClassifier(
    max_depth=2,
    class_weight="balanced",
    random_state=42,
)
tree.fit(X_train, y_train)

# Print tree
tree_text = export_text(tree, feature_names=MODEL_NUMERIC_FEATURES)
print("  Decision tree rules:")
print("-" * 60)
print(tree_text)
print("-" * 60)

# Tree scores on test set
tree_proba = tree.predict_proba(X_test)[:, 1]

# Hand-rule scores on test set
hr_test_scores = X_test.index.map(
    df.loc[X_test.index, "hand_rule_score"]
).values

# Precision@Key
def precision_at_k(scores, labels, k):
    """Calculate precision at k using descending score ranking."""
    order = np.argsort(-np.asarray(scores))
    top_k = order[:k]
    return float(labels.iloc[top_k].mean())

precision_results = {}
print("\n  Precision@K comparison (test clients):")
print("  " + "-" * 50)
print(f"  {'Metric':<35} {'Hand rule':>10} {'Tree(depth2)':>12}")
print("  " + "-" * 50)
for k in [20, 50]:
    hr = precision_at_k(hr_test_scores, y_test, k)
    tr = precision_at_k(tree_proba, y_test, k)
    precision_results[str(k)] = {"hand_rule": hr, "tree_depth_2": tr}
    print(f"  Precision@{k:<28} {hr:>10.3f} {tr:>12.3f}")

# Depth-3 for comparison
tree3 = DecisionTreeClassifier(
    max_depth=3,
    class_weight="balanced",
    random_state=42,
)
tree3.fit(X_train, y_train)
tree3_proba = tree3.predict_proba(X_test)[:, 1]

print("  " + "-" * 50)
print("  Precision@50:")
print(f"    hand rule       = {precision_at_k(hr_test_scores, y_test, 50):.3f}")
print(f"    depth-2 tree    = {precision_at_k(tree_proba, y_test, 50):.3f}")
print(f"    depth-3 tree    = {precision_at_k(tree3_proba, y_test, 50):.3f}")
print(f"  Tree (depth 2): {tree.tree_.node_count} nodes")
print(f"  Tree (depth 3): {tree3.tree_.node_count} nodes")


# Hand-rule scores on test set

# ---------------------------------------------------------------------------
# Task 3 — Leakage check
# ---------------------------------------------------------------------------
print("\n[TASK 3] Leakage check")
print("  Verify that outcome columns (trend_direction, trend_pct) are NOT used")
print("  as model features.")

used_features = set(MODEL_NUMERIC_FEATURES) & set(LEAKAGE_COLUMNS)
leakage_found = len(used_features) > 0

print(f"  Leakage columns: {LEAKAGE_COLUMNS}")
print(f"  Features used: {sorted(MODEL_NUMERIC_FEATURES)}")
print(f"  Leakage columns used as features: {sorted(used_features) if leakage_found else 'NONE — OK'}")

label_in_features = "is_declining_label" in MODEL_NUMERIC_FEATURES
print(f"  Target variable in features: {'YES — LEAKAGE!' if label_in_features else 'NO — OK'}")

print("\n  Leakage conclusion:",
      "FAIL — outcome leaked into features" if leakage_found else "PASS — outcome is not a feature")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "week": 2,
    "assignment": "The model is just a rule you can read",
    "summary": {
        "n_pages": int(n_pages),
        "decline_rate": float(decline_rate),
        "stale_days": STALE_DAYS,
        "min_impressions": MIN_IMPRESSIONS,
        "flagged_pages": int((df["hand_rule_score"] > 0).sum()),
        "tree_depth": 2,
        "tree_nodes": int(tree.tree_.node_count),
        "leakage_found": bool(leakage_found),
    },
    "precision_at_k": {
        "hand_rule": {k: precision_results[k]["hand_rule"] for k in ["20", "50"]},
        "tree_depth_2": {k: precision_results[k]["tree_depth_2"] for k in ["20", "50"]},
        "tree_depth_3": {
            "20": float(precision_at_k(tree3_proba, y_test, 20)),
            "50": float(precision_at_k(tree3_proba, y_test, 50)),
        },
    },
    "decision_tree_rules": tree_text,
    "top_hand_rule_pages": df_sorted[
        ["content_id", "days_since_last_update", "impressions_90d", "hand_rule_score"]
    ].head(5).to_dict(orient="records"),
}

with open(RESULTS_JSON_PATH, "w") as f:
    json.dump(results, f, indent=2)

# Build Markdown report
report_lines = [
    "# Week 2 — The model is just a rule you can read\n",
    "FlyRank ML Internship · Applied Search Intelligence track\n",
    "\n",
    "## Summary\n",
    "\n",
    f"- Pages analysed: {n_pages:,}\n",
    f"- Declining rate: {decline_rate:.1%}\n",
    f"- Hand-rule parameters: STALE_DAYS={STALE_DAYS}, MIN_IMPRESSIONS={MIN_IMPRESSIONS}\n",
    f"- Pages flagged 'stale x visible': {(df['hand_rule_score'] > 0).sum():,}\n",
    f"- Decision tree (depth 2): {tree.tree_.node_count} nodes\n",
    f"- Leakage check: {'FAIL' if leakage_found else 'PASS'}\n",
    "\n",
    "## Results\n",
    "\n",
    "### Precision @ K (test client split, 20% holdout)\n",
    "\n",
    "| Metric | k=20 | k=50 |\n",
    "|--------|------|------|\n",
    f"| Hand rule | {precision_results['20']['hand_rule']:.3f} | {precision_results['50']['hand_rule']:.3f} |\n",
    f"| Decision tree (depth 2) | {precision_results['20']['tree_depth_2']:.3f} | {precision_results['50']['tree_depth_2']:.3f} |\n",
    f"| Decision tree (depth 3) | {results['precision_at_k']['tree_depth_3']['20']:.3f} | {results['precision_at_k']['tree_depth_3']['50']:.3f} |\n",
    "\n",
    "### Decision Tree (depth 2) — Readable if/else rules\n",
    "\n",
    "```\n",
    f"{tree_text.replace(chr(10), chr(10) + '    ')}```\n",
    "\n",
    "### Top pages flagged by hand rule (stale x visible)\n",
    "\n",
    "| content_id | days_since_last_update | impressions_90d | hand_rule_score |\n",
    "|---|---|---|---|\n",
]
for p in results["top_hand_rule_pages"]:
    report_lines.append(
        f"| {p['content_id']} | {p['days_since_last_update']} | {p['impressions_90d']} | {p['hand_rule_score']} |\n"
    )

report_lines += [
    "\n",
    "## Task Checklist\n",
    "\n",
    "- [x] **Setup**: Load data, define `is_declining_label` from `trend_direction`\n",
    "- [x] **Task 1**: Hand rule `stale x visible` implemented and ranked\n",
    "- [x] **Task 2**: Depth-2 decision tree fitted, printed, and compared\n",
    "- [x] **Task 3**: Leakage check passed — outcome not used as a feature\n",
    "\n",
    "## Key Takeaways\n",
    "\n",
    "1. The hand rule `stale x visible` is a transparent, explainable baseline.\n",
    "2. The decision tree is also readable: `if impressions <= X then ... else ...`.\n",
    "3. Depth-3 trees capture more signal but are harder to read — the trade-off\n",
    "   between interpretability and performance.\n",
    "4. **Never feed the outcome back in** — `trend_direction` and `trend_pct` must\n",
    "   stay out of the feature list, or the model is learning from leaked labels.\n",
    "\n",
    "---\n",
    "*Generated by `week2_ml_internship.py`*\n",
]

with open(OUTPUT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nResults saved to: {OUTPUT_PATH}")
print(f"Results JSON saved to: {RESULTS_JSON_PATH}")
print("\n" + "=" * 60)
print("WEEK 2 COMPLETE")
print("=" * 60)

hr_test_scores = X_test.index.map(
    df.loc[X_test.index, "hand_rule_score"]
).values

# Precision@Key
def precision_at_k(scores, labels, k):
    """Calculate precision at k using descending score ranking."""
    order = np.argsort(-np.asarray(scores))
    top_k = order[:k]
    return float(labels.iloc[top_k].mean())


print("\n  Precision@K comparison (test clients):")
print("  " + "-" * 50)
print(f"  {'Metric':<35} {'Hand rule':>10} {'Tree(depth2)':>12}")
print("  " + "-" * 50)
precision_results = {}
for k in [20, 50]:
    hr = precision_at_k(hr_test_scores, y_test, k)
    tr = precision_at_k(tree_proba, y_test, k)
    precision_results[str(k)] = {"hand_rule": hr, "tree_depth_2": tr}
    print(f"  Precision@{k:<28} {hr:>10.3f} {tr:>12.3f}")

# Depth-3 for comparison
tree3 = DecisionTreeClassifier(
    max_depth=3,
    class_weight="balanced",
    random_state=42,
)
tree3.fit(X_train, y_train)
tree3_proba = tree3.predict_proba(X_test)[:, 1]

print("  " + "-" * 50)
print("  Precision@50:")
print(f"    hand rule       = {precision_at_k(hr_test_scores, y_test, 50):.3f}")
print(f"    depth-2 tree    = {precision_at_k(tree_proba, y_test, 50):.3f}")
print(f"    depth-3 tree    = {precision_at_k(tree3_proba, y_test, 50):.3f}")
print(f"  Tree (depth 2): {tree.tree_.node_count} nodes")
print(f"  Tree (depth 3): {tree3.tree_.node_count} nodes")

print(df_sorted[["content_id", "days_since_last_update", "impressions_90d", "hand_rule_score"]].head(5).to_string(index=False))
