import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("e:/Bdtorrent/CHikni/data/raw/content_refresh_anonymized.csv")
df["is_declining_label"] = (df["trend_direction"].str.lower().eq("down")).astype(int)
HONEST_FEATURES = ["impressions_90d","clicks_90d","sessions_90d","days_since_last_update","content_age_days","ctr","avg_position","engagement_rate","scroll_rate"]
X = df[HONEST_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["is_declining_label"].values

SEALED_CLIENT = sorted(df["client_id"].unique())[0]
sealed_mask = df["client_id"] == SEALED_CLIENT
train_mask = ~sealed_mask

sm = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
sm.fit(X[train_mask], y[train_mask])
sp = sm.predict_proba(X[sealed_mask])[:,1]

def pak(scores, labels, k):
    o = np.argsort(-np.asarray(scores))
    return float(np.asarray(labels)[o[:min(k,len(labels))]].mean())

K_VALUES = [10,20,50,100]
sealed_metrics = {
    "client": SEALED_CLIENT,
    "n_sealed": int(sealed_mask.sum()),
    "sealed_base_rate": float(y[sealed_mask].mean()),
    "split": "sealed_holdout_grouped_by_client",
    "model": "random_forest",
    "features": HONEST_FEATURES,
    "note": "One entire client held out and never seen in training. Frame built by the sealed-holdout cell in ml09_hunt_leakage_validate.ipynb.",
}
for k in K_VALUES:
    sealed_metrics["precision_at_%d" % k] = round(pak(sp, y[sealed_mask], k), 4)

out_dir = Path("e:/Bdtorrent/CHikni/work/notebooks")
out_dir.mkdir(parents=True, exist_ok=True)
metrics_path = out_dir / "ml09_sealed_holdout_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(sealed_metrics, f, indent=2)
print("Wrote", metrics_path)
print(json.dumps(sealed_metrics, indent=2))