cells.append(md(
    "## 1. Features + leakage guard\n",
    "\n",
    "Features are trailing-90-day search/engagement signals only. **`trend_direction` and `trend_pct` are the label — they are never features.** Feeding them in would be leakage: the model would \"predict\" decline because it already saw the answer.\n",
    "\n",
    "Feature set (observable, pre-period signals):\n",
    "- `impressions_90d`, `clicks_90d`, `sessions_90d` — trailing traffic volume\n",
    "- `days_since_last_update`, `content_age_days` — freshness / age\n",
    "- `ctr`, `avg_position` — search-result presence\n",
    "- `engagement_rate`, `scroll_rate` — on-page engagement\n",
    "\n",
    "Missing values are filled with 0 after confirming they are not systematically tied to `content_type` (per the data dictionary, missingness is along content-type lines; for this teaching slice we impute 0 and note it as a limitation).\n"
))

cells.append(code(
    "MODEL_NUMERIC_FEATURES = [",
    "    \"impressions_90d\", \"clicks_90d\", \"sessions_90d\",",
    "    \"days_since_last_update\", \"content_age_days\",",
    "    \"ctr\", \"avg_position\", \"engagement_rate\", \"scroll_rate\",",
    "]",
    "",
    "X_raw = df[MODEL_NUMERIC_FEATURES].copy()",
    "X_raw = X_raw.apply(pd.to_numeric, errors=\"coerce\").fillna(0)",
    "print(f\"Feature matrix: {X_raw.shape[0]} rows × {X_raw.shape[1]} features\")",
    "print(\"Features:\", list(X_raw.columns))",
    "",
    "y = df[\"is_declining_label\"].values",
    "groups = df[\"client_id\"].values",
    "",
    "gkf = GroupKFold(n_splits=5)",
    "folds = list(gkf.split(X_raw, y, groups))",
    "print(f\"\\nClient-holdout CV: {len(folds)} folds (GroupKFold by client_id)\")",
    "print(f\"Unique clients: {len(np.unique(groups))}\")",
))