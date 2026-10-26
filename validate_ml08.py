import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

print("sklearn", sklearn.__version__)
df = pd.read_csv("e:/Bdtorrent/CHikni/data/raw/content_refresh_anonymized.csv")
df["is_declining_label"] = (df["trend_direction"].str.lower().eq("down")).astype(int)
base_rate = float(df["is_declining_label"].mean())
print("shape", df.shape, "clients", df["client_id"].nunique(), "base_rate", round(base_rate, 4))

FEAT = ["impressions_90d", "clicks_90d", "sessions_90d", "days_since_last_update",
        "content_age_days", "ctr", "avg_position", "engagement_rate", "scroll_rate"]
X = df[FEAT].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["is_declining_label"].values
groups = df["client_id"].values

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups))
print("folds", len(folds))

def pak(scores, labels, k):
    o = np.argsort(-np.asarray(scores))
    return float(np.asarray(labels)[o[:min(k, len(labels))]].mean())

K = [10, 20, 50, 100]
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
models = {"logistic_regression": lr, "random_forest": rf}
oof = {n: np.zeros(len(X)) for n in models}

for fi, (tr, te) in enumerate(folds):
    for n, m in models.items():
        m.fit(X.iloc[tr], y[tr])
        oof[n][te] = m.predict_proba(X.iloc[te])[:, 1]

STALE = 180
MINIMP = 500
df["is_stale"] = (df["days_since_last_update"] >= STALE).astype(int)
df["is_visible"] = (df["impressions_90d"] >= MINIMP).astype(int)
df["queue_score"] = (df["is_stale"] * df["is_visible"] * df["impressions_90d"]).astype(float)

print("\nk   baseline   LR        RF        base")
for k in K:
    print(f"{k:3}  {pak(df['queue_score'], y, k):.4f}   {pak(oof['logistic_regression'], y, k):.4f}   {pak(oof['random_forest'], y, k):.4f}   {base_rate:.4f}")

bm = max(models, key=lambda n: pak(oof[n], y, 50))
print("\nBest @50:", bm, round(pak(oof[bm], y, 50), 4), "baseline:", round(pak(df["queue_score"], y, 50), 4), "base_rate:", round(base_rate, 4))

rf_all = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
rf_all.fit(X, y)
imp = pd.DataFrame({"feature": X.columns, "importance": rf_all.feature_importances_}).sort_values("importance", ascending=False)
print("\nRF feature importances:")
print(imp.to_string(index=False))

print("\nOK - ML-08 logic runs cleanly")