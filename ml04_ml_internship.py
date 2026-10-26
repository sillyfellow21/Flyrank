#!/usr/bin/env python3
import json
from pathlib import Path

MODEL_NUMERIC_FEATURES = [
    "impressions_total",
    "clicks_total",
    "april_impressions",
    "april_clicks",
    "feb_clicks",
    "momentum",
    "ctr",
    "active_days",
    "weighted_position",
]

ELIGIBILITY_IMPRESSIONS = 1000
ELIGIBILITY_CLICKS = 10

K_VALUES = [10, 20, 50, 100]

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
# 1. Setup
print("\n[1] Setup: Loading data...")
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Data file not found: " + str(DATA_PATH) + "\n"
        "Expected: data/raw/content_refresh_anonymized.csv"
    )

df = pd.read_csv(DATA_PATH)
print("  Loaded {:,} pages x {}".format(len(df), df.shape[1]))

if "trend_direction" in df.columns:
    df["is_declining_label"] = df["trend_direction"].str.lower().eq("down").astype(int)
else:
    df["is_declining_label"] = df["is_declining"].astype(int)

n_pages = df.shape[0]
decline_rate = df["is_declining_label"].mean()
print("  Pages analysed: {:,}".format(n_pages))
print("  Declining rate: {:.1%}".format(decline_rate))

    "min_samples_leaf": 5,
# 2. Feature Engineering
print("\n[2] Feature engineering...")

df["impressions_total"] = df["impressions_90d"].fillna(0)
df["clicks_total"] = df["clicks_90d"].fillna(0)
df["april_impressions"] = df["impressions_last_30d"].fillna(0)
df["april_clicks"] = df["clicks_last_30d"].fillna(0)
df["feb_clicks"] = df["clicks_prev_30d"].fillna(0)

with np.errstate(divide="ignore", invalid="ignore"):
    df["momentum"] = (df["impressions_90d"] - df["impressions_prev_30d"]) / df["impressions_prev_30d"].replace(0, np.nan) * 100
df["momentum"] = df["momentum"].fillna(0).clip(-100, 500)

df["ctr"] = np.where(df["impressions_90d"] > 0, df["clicks_90d"] / df["impressions_90d"] * 100, 0)

df["active_days"] = df["days_with_impressions"].fillna(0)

if "avg_position" in df.columns:
    df["weighted_position"] = df["avg_position"].fillna(0)
else:
    impression_tier_map = {"low": 1.0, "low-mid": 0.75, "mid": 0.5, "high-mid": 0.25, "high": 0.1}
    df["weighted_position"] = df.get("impression_tier", pd.Series(dtype=float)).map(impression_tier_map).fillna(0)

missing_cols = [c for c in MODEL_NUMERIC_FEATURES if c not in df.columns]
if missing_cols:
    print("  Warning: Creating missing features:", missing_cols)
    for col in missing_cols:
        df[col] = 0.0

# 4. Baseline: Hand-rule score
print("\n[4] Baseline hand-rule scoring...")

STALE_DAYS = 90
MIN_IMPRESSIONS = 100

df_eligible["hand_rule_score"] = np.where(
    (df_eligible["days_since_last_update"] > STALE_DAYS)
    & (df_eligible["impressions_90d"] >= MIN_IMPRESSIONS),
    df_eligible["impressions_90d"],
    0
)

print("  Hand-rule parameters: STALE_DAYS={}, MIN_IMPRESSIONS={}".format(STALE_DAYS, MIN_IMPRESSIONS))
print("  Pages flagged by hand rule: {:,}".format((df_eligible['hand_rule_score'] > 0).sum()))

# 5. Model: Random Forest with GroupKFold by client
print("\n[5] Training Random Forest with GroupKFold (by client)...")

y = df_eligible["is_declining_label"].values
groups = df_eligible["client_id"].values

unique_clients = np.unique(groups)
print("  Unique clients for CV: {}".format(len(unique_clients)))

group_kfold = GroupKFold(n_splits=5)
folds = list(group_kfold.split(X, y, groups))

print("  Folds: {} (GroupKFold by client)".format(len(folds)))

oof_proba = np.zeros(len(X))

for fold_idx, (train_idx, test_idx) in enumerate(folds):
    print("  Fold {}/{}...".format(fold_idx + 1, len(folds)))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    oof_proba[test_idx] = rf.predict_proba(X_test)[:, 1]

    fold_precision = {}
    for k in K_VALUES:
        if len(test_idx) >= k:
            order = np.argsort(-oof_proba[test_idx])
            top_k = order[:k]
            fold_precision[k] = float(y_test[top_k].mean())
        else:
            fold_precision[k] = 0.0

    print("    Fold {} Precision: @10={:.3f} @20={:.3f} @50={:.3f} @100={:.3f}".format(
        fold_idx + 1, fold_precision[10], fold_precision[20], fold_precision[50], fold_precision[100]))

# Fit final model on all data for feature importances
rf = RandomForestClassifier(**RF_PARAMS)
rf.fit(X, y)

feature_importances = dict(zip(MODEL_NUMERIC_FEATURES, rf.feature_importances_))
feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

X = df[MODEL_NUMERIC_FEATURES].copy()
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# 3. Eligibility Filter
print("\n[3] Applying eligibility criteria...")

eligibility_mask = (
    (df["impressions_total"] >= ELIGIBILITY_IMPRESSIONS)
    & (df["april_clicks"] >= ELIGIBILITY_CLICKS)
)
eligible_mask = eligibility_mask & df["is_declining_label"].notna()

df_eligible = df[eligibility_mask].copy()
df_eligible["is_declining_label"] = df_eligible["is_declining_label"].astype(int)

n_eligible = df_eligible.shape[0]
decline_rate_eligible = df_eligible["is_declining_label"].mean()

print("  Eligible pages: {:,} (from {:,} total)".format(n_eligible, n_pages))
print("  Declining rate (eligible): {:.1%}".format(decline_rate_eligible))
print("  Unique clients: {}".format(df_eligible['client_id'].nunique()))

    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

print("=" * 70)
print("ML-04 - Prioritizing Search-Performance Decline for Human Review")
print("=" * 70)
