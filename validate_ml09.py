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

HONEST_FEATURES = ["impressions_90d","clicks_90d","sessions_90d","days_since_last_update","content_age_days","ctr","avg_position","engagement_rate","scroll_rate"]
X_honest = df[HONEST_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["is_declining_label"].values
groups = df["client_id"].values

def pak(scores, labels, k):
    o = np.argsort(-np.asarray(scores))
    return float(np.asarray(labels)[o[:min(k,len(labels))]].mean())

K_VALUES = [10,20,50,100]
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_honest, y, groups))

def cv_pak(Xdf, model_name="random_forest", k=50):
    oof = np.zeros(len(Xdf))
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    model = rf if model_name=="random_forest" else lr
    for tr, te in folds:
        model.fit(Xdf.iloc[tr], y[tr])
        oof[te] = model.predict_proba(Xdf.iloc[te])[:,1]
    return oof

# 2a label-derived leakage
X_leaky_label = X_honest.copy()
X_leaky_label["trend_direction_enc"] = (df["trend_direction"].str.lower().eq("down")).astype(int)
X_leaky_label["trend_pct"] = pd.to_numeric(df["trend_pct"], errors="coerce").fillna(0)
oof_honest = cv_pak(X_honest, "random_forest", 50)
oof_leaky_label = cv_pak(X_leaky_label, "random_forest", 50)
print("2a label-derived: honest=%.4f leaky=%.4f (Δ%+.4f)" % (pak(oof_honest,y,50), pak(oof_leaky_label,y,50), pak(oof_leaky_label,y,50)-pak(oof_honest,y,50)))

# 2b product flags leakage
product_flags = ["health_score","is_underperformer","is_quick_win","needs_ctr_fix","needs_engagement_fix","needs_indexing","ai_opportunity","is_initial_refresh_candidate"]
X_leaky_flags = X_honest.copy()
for flag in product_flags:
    if flag in df.columns:
        X_leaky_flags[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0)
oof_leaky_flags = cv_pak(X_leaky_flags, "random_forest", 50)
print("2b product flags:   honest=%.4f leaky=%.4f (Δ%+.4f)" % (pak(oof_honest,y,50), pak(oof_leaky_flags,y,50), pak(oof_leaky_flags,y,50)-pak(oof_honest,y,50)))

# 2c overlapping window leakage
X_leaky_window = X_honest.copy()
X_leaky_window["impressions_last_30d"] = pd.to_numeric(df["impressions_last_30d"], errors="coerce").fillna(0)
X_leaky_window["clicks_last_30d"] = pd.to_numeric(df["clicks_last_30d"], errors="coerce").fillna(0)
oof_leaky_window = cv_pak(X_leaky_window, "random_forest", 50)
print("2c overlapping win: honest=%.4f leaky=%.4f (Δ%+.4f)" % (pak(oof_honest,y,50), pak(oof_leaky_window,y,50), pak(oof_leaky_window,y,50)-pak(oof_honest,y,50)))
print("base rate: %.4f" % base_rate)

# grouped vs random split
oof_grouped = {"logistic_regression":np.zeros(len(X_honest)), "random_forest":np.zeros(len(X_honest))}
for tr,te in folds:
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    for name,model in [("logistic_regression",lr),("random_forest",rf)]:
        model.fit(X_honest.iloc[tr], y[tr])
        oof_grouped[name][te] = model.predict_proba(X_honest.iloc[te])[:,1]

rng = np.random.RandomState(42)
perm = rng.permutation(len(X_honest)); cut=int(0.8*len(X_honest)); tr_r,te_r=perm[:cut],perm[cut:]
random_results={}
for name in ["logistic_regression","random_forest"]:
    if name=="logistic_regression":
        m=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    else:
        m=RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    m.fit(X_honest.iloc[tr_r], y[tr_r])
    pr=m.predict_proba(X_honest.iloc[te_r])[:,1]
    random_results[name]=[pak(pr,y[te_r],k) for k in K_VALUES]

print("\nmodel           K10   K20   K50   K100  (grouped)")
for name in ["logistic_regression","random_forest"]:
    gs=[pak(oof_grouped[name],y,k) for k in K_VALUES]
    rs=random_results[name]
    gap=[rs[i]-gs[i] for i in range(4)]
    print("%-16s %.3f  %.3f  %.3f  %.3f   gap:%s" % (name, gs[0],gs[1],gs[2],gs[3], ["%+.3f"%g for g in gap]))

# sealed holdout
SEALED_CLIENT=sorted(df["client_id"].unique())[0]
sealed_mask=df["client_id"]==SEALED_CLIENT
train_mask=~sealed_mask
sm=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=-1)
sm.fit(X_honest[train_mask], y[train_mask])
sp=sm.predict_proba(X_honest[sealed_mask])[:,1]
print("\nSealed holdout client:", SEALED_CLIENT, "n_sealed:", int(sealed_mask.sum()), "sealed_base_rate: %.4f" % float(y[sealed_mask].mean()))
for k in K_VALUES:
    print("  precision@%d: %.4f" % (k, pak(sp, y[sealed_mask], k)))

# feature importances
rf_all=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=-1)
rf_all.fit(X_honest,y)
imp=pd.DataFrame({"feature":X_honest.columns,"importance":rf_all.feature_importances_}).sort_values("importance",ascending=False)
print("\nTop features:"); print(imp.head(5).to_string(index=False))
print("\nOK - ML-09 logic runs cleanly")