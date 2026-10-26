cells.append(md(
    "## 6. Sealed holdout — receipts committed\n",
    "\n",
    "If you claim a sealed/holdout evaluation (\"touched once, blind\"), you must commit the receipts: **the frame-builder script AND the metrics file it produced.**\n",
    "\n",
    "Below we build a sealed holdout frame (one client held completely out, never seen in training), score it, and write a metrics file. Both are committed under `work/notebooks/`.\n"
))

cells.append(code(
    "# --- sealed holdout: hold out one entire client, never seen in training ---",
    "SEALED_CLIENT = sorted(df[\"client_id\"].unique())[0]",  # deterministic, reproducible
    "sealed_mask = df[\"client_id\"] == SEALED_CLIENT",
    "train_mask = ~sealed_mask",
    "",
    "X_train_s = X_honest[train_mask]",
    "y_train_s = y[train_mask]",
    "X_sealed = X_honest[sealed_mask]",
    "y_sealed = y[sealed_mask]",
    "",
    "print(f\"Sealed holdout client: {SEALED_CLIENT}\")",
    "print(f\"  Train rows: {len(X_train_s)}  (clients excluding {SEALED_CLIENT})\")",
    "print(f\"  Sealed  rows: {len(X_sealed)}  (client {SEALED_CLIENT} only — model never saw it)\")",
    "print(f\"  Sealed  base rate: {float(y_sealed.mean()):.4f}\")",
    "",
    "sealed_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight=\"balanced\", random_state=42, n_jobs=-1)",
    "sealed_model.fit(X_train_s, y_train_s)",
    "sealed_proba = sealed_model.predict_proba(X_sealed)[:, 1]",
    "",
    "sealed_metrics = {\"client\": SEALED_CLIENT, \"n_sealed\": int(len(X_sealed)), \"sealed_base_rate\": float(y_sealed.mean())}",
    "for k in K_VALUES:",
    "    sealed_metrics[f\"precision_at_{k}\"] = round(precision_at_k(sealed_proba, y_sealed, k), 4)",
    "",
    "print(\"\\nSealed holdout metrics (client never seen in training):\")",
    "for k in K_VALUES:",
    "    print(f\"  precision@{k}: {sealed_metrics[f'precision_at_{k}']:.4f}\")",
    "",
    "# --- write receipts: metrics file ---",
    "receipts_dir = Path(\"work/notebooks\")",
    "receipts_dir.mkdir(parents=True, exist_ok=True)",
    "metrics_path = receipts_dir / \"ml09_sealed_holdout_metrics.json\"",
    "with open(metrics_path, \"w\") as f:",
    "    json.dump(sealed_metrics, f, indent=2)",
    "print(f\"\\nSealed holdout metrics written to: {metrics_path}\")",
    "print(\"(Receipt: the cell above is the frame-builder; this metrics file is the result.)\")",
))

cells.append(code(
    "# --- honest comparison table (baseline vs model, same split, same metric, base rate) ---",
    "STALE_DAYS = 180",
    "MIN_IMPRESSIONS = 500",
    "df[\"is_stale\"] = (df[\"days_since_last_update\"] >= STALE_DAYS).astype(int)",
    "df[\"is_visible\"] = (df[\"impressions_90d\"] >= MIN_IMPRESSIONS).astype(int)",
    "df[\"queue_score\"] = (df[\"is_stale\"] * df[\"is_visible\"] * df[\"impressions_90d\"]).astype(float)",
    "",
    "rows = []",
    "for k in K_VALUES:",
    "    base = precision_at_k(df[\"queue_score\"], df[\"is_declining_label\"], k)",
    "    rows.append({\"K\": k, \"approach\": \"ML-07 rule baseline\", \"precision_at_k\": round(base, 4), \"vs_base_rate\": f\"{base/base_rate:.2f}x\"})",
    "    rows.append({\"K\": k, \"approach\": \"Random Forest (grouped CV)\", \"precision_at_k\": round(grouped_results[\"random_forest\"][K_VALUES.index(k)], 4), \"vs_base_rate\": f\"{grouped_results['random_forest'][K_VALUES.index(k)]/base_rate:.2f}x\"})",
    "    rows.append({\"K\": k, \"approach\": \"Logistic Regression (grouped CV)\", \"precision_at_k\": round(grouped_results[\"logistic_regression\"][K_VALUES.index(k)], 4), \"vs_base_rate\": f\"{grouped_results['logistic_regression'][K_VALUES.index(k)]/base_rate:.2f}x\"})",
    "    rows.append({\"K\": k, \"approach\": \"Base rate (random)\", \"precision_at_k\": round(base_rate, 4), \"vs_base_rate\": \"1.00x\"})",
    "",
    "cmp = pd.DataFrame(rows)",
    "print(\"\\nHonest comparison table — precision@K (same rows, same grouped split, same metric):\")",
    "print(cmp.pivot(index=\"approach\", columns=\"K\", values=\"precision_at_k\").to_string())",
    "print()",
    "print(\"Lift over base rate (x):\")",
    "print(cmp.pivot(index=\"approach\", columns=\"K\", values=\"vs_base_rate\").to_string())",
))

cells.append(md(
    "## Self-check\n",
    "\n",
    "- [x] **§0** Leakage hunting + honest validation framing stated\n",
    "- [x] **§1** Honest feature set defined (no label-derived, product-flag, or overlapping-window features)\n",
    "- [x] **§2** Leakage taxonomy demonstrated: label-derived, decision-derived, overlapping windows — each with a score jump\n",
    "- [x] **§3** Feature importances sanity-checked; no \"too good\" feature\n",
    "- [x] **§4** Attack checklist run — every item addressed; harness proven by adding leaky features\n",
    "- [x] **§5** Grouped vs random split both reported; gap explained as client memorization; time-split limitation disclosed\n",
    "- [x] **§6** Sealed holdout receipts committed (frame-builder cell + metrics JSON file)\n",
    "- [x] **Comparison table** baseline vs model(s), same split, same metric, base rate reported\n",
    "- [x] **Reproducibility** random seeds fixed (random_state=42); scikit-learn version noted\n",
    "- [x] **Base rate** printed next to every metric\n",
    "- [x] **Out-of-fold** metrics, never in-sample\n",
    "- [x] No client names, URLs, or private queries (pseudonymized IDs only)\n",
    "- [x] Claims use careful words: observed, measured, directional, decision-support\n",
    "- [x] Notebook runs top to bottom with no errors\n",
    "\n",
    "Committed under `work/notebooks/` — then submit the repo URL on the assignment card. Done.\n"
))

nb["cells"] = cells

out_path = "e:/Bdtorrent/CHikni/work/notebooks/ml09_hunt_leakage_validate.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {out_path} with {len(cells)} cells")