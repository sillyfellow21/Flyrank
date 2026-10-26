cells.append(md(
    "### 2c. Future / overlapping windows\n",
    "\n",
    "A feature summed over a window that **CONTAINS the label's window** already knows the outcome. Draw the timeline: every feature must be knowable at the moment of prediction, labels strictly after.\n",
    "\n",
    "Here the label (`trend_direction`) is computed over the trailing 90 days. `impressions_last_30d` and `clicks_last_30d` are sub-aggregates **inside** that same 90-day window — they overlap the label period. Only the *earlier* sub-window (`impressions_prev_30d`) is a legal feature for a label defined on the final month; the trailing aggregates are tainted.\n",
    "\n",
    "For this teaching snapshot we treat `impressions_90d`/`clicks_90d` as the trailing baseline window (consistent with ML-07/ML-08), and we **exclude** the `_last_30d` columns from the honest feature set. In the warehouse release (weeks 3+), you would align feature and label windows explicitly and use only `*_prev30` columns.\n",
))

cells.append(code(
    "# --- 2c: overlapping-window leakage demo (impressions_last_30d inside label window) ---",
    "X_leaky_window = X_honest.copy()",
    "X_leaky_window[\"impressions_last_30d\"] = pd.to_numeric(df[\"impressions_last_30d\"], errors=\"coerce\").fillna(0)",
    "X_leaky_window[\"clicks_last_30d\"] = pd.to_numeric(df[\"clicks_last_30d\"], errors=\"coerce\").fillna(0)",
    "",
    "oof_leaky_window = cv_precision_at_k(X_leaky_window, y, groups, \"random_forest\", 50)",
    "p_leaky_window = precision_at_k(oof_leaky_window, y, 50)",
    "",
    "print(\"2c — Overlapping-window leakage (impressions/clicks_last_30d inside 90-day label window):\")",
    "print(f\"  Honest model precision@50 : {p_honest:.4f}\")",
    "print(f\"  Leaky  model precision@50 : {p_leaky_window:.4f}\")",
    "print(f\"  Inflation (leak→honest)   : {p_leaky_window - p_honest:+.4f}\")",
    "print(f\"  Base rate                 : {base_rate:.4f}\")",
    "",
    "print(\"\\nSummary of leakage demos (precision@50, random_forest, grouped CV):\")",
    "print(f\"  honest           : {p_honest:.4f}\")",
    "print(f\"  + label-derived  : {p_leaky_label:.4f}  (confession: near-perfect) \")",
    "print(f\"  + product flags  : {p_leaky_flags:.4f}  (circular) \")",
    "print(f\"  + overlapping win: {p_leaky_window:.4f}  (tainted window) \")",
    "print(f\"  base rate        : {base_rate:.4f}\")",
))

cells.append(md(
    "## 3. Feature importance sanity check\n",
    "\n",
    "After training, we ask what the model leans on — then sanity-check: is the top feature plausibly related to decline, or is it suspiciously perfect (a sign of leakage)? We investigate \"too good\", never celebrate it.\n",
))

cells.append(code(
    "# --- fit RF on all data for feature importances (honest features only) ---",
    "rf_all = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight=\"balanced\", random_state=42, n_jobs=-1)",
    "rf_all.fit(X_honest, y)",
    "",
    "imp = pd.DataFrame({",
    "    \"feature\": X_honest.columns,",
    "    \"importance\": rf_all.feature_importances_,",
    "}).sort_values(\"importance\", ascending=False)",
    "imp[\"importance\"] = imp[\"importance\"].round(4)",
    "print(\"\\nHonest Random Forest feature importances (fit on all data):\")",
    "print(imp.to_string(index=False))",
    "",
    "top_feat = imp.iloc[0][\"feature\"]",
    "top_imp = float(imp.iloc[0][\"importance\"])",
    "print(f\"\\nTop feature: {top_feat} (importance {top_imp:.4f})\")",
    "print(\"Sanity: impressions_90d / age / position are plausible decline signals — not suspiciously perfect.\")",
    "print(\"If a feature towered above all others (~0.8+) with near-perfect precision, that would be a leakage confession.\")",
))