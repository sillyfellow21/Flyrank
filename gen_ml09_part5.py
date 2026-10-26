cells.append(md(
    "## 5. Honest validation — grouped split vs random split\n",
    "\n",
    "Random is rarely honest: rows from one client share hidden character, and a random split lets the model memorize the client and fake skill. The honest question is \"does it work on a client it never saw?\"\n",
    "\n",
    "We report **both** numbers — the gap between them is itself a finding about how much memorization was happening.\n"
))

cells.append(code(
    "# --- grouped (client-holdout) CV ---",
    "print(\"\\n=== GROUPED (client-holdout) CV — honest ===\")",
    "oof_grouped = {name: np.zeros(len(X_honest)) for name in [\"logistic_regression\", \"random_forest\"]}",
    "for fold_idx, (tr, te) in enumerate(folds):",
    "    X_tr, X_te = X_honest.iloc[tr], X_honest.iloc[te]",
    "    y_tr, y_te = y[tr], y[te]",
    "    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=\"balanced\", random_state=42))",
    "    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight=\"balanced\", random_state=42, n_jobs=-1)",
    "    for name, model in [(\"logistic_regression\", lr), (\"random_forest\", rf)]:",
    "        model.fit(X_tr, y_tr)",
    "        oof_grouped[name][te] = model.predict_proba(X_te)[:, 1]",
    "",
    "print(f\"{'model':<20} \" + \" \".join(f\"@K={k:>3}\" for k in K_VALUES))",
    "print(\"-\" * 60)",
    "grouped_results = {}",
    "for name in [\"logistic_regression\", \"random_forest\"]:",
    "    scores = [precision_at_k(oof_grouped[name], y, k) for k in K_VALUES]",
    "    grouped_results[name] = scores",
    "    print(f\"{name:<20} \" + \" \".join(f\"{s:>7.3f}\" for s in scores))",
))

cells.append(code(
    "# --- random split (dishonest comparison) ---",
    "print(\"\\n=== RANDOM split — dishonest (rows from same client in train & test) ===\")",
    "rng = np.random.RandomState(42)",
    "perm = rng.permutation(len(X_honest))",
    "cut = int(0.8 * len(X_honest))",
    "tr_r, te_r = perm[:cut], perm[cut:]",
    "",
    "random_results = {}",
    "for name in [\"logistic_regression\", \"random_forest\"]:",
    "    if name == \"logistic_regression\":",
    "        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=\"balanced\", random_state=42))",
    "    else:",
    "        m = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, class_weight=\"balanced\", random_state=42, n_jobs=-1)",
    "    m.fit(X_honest.iloc[tr_r], y[tr_r])",
    "    proba_r = m.predict_proba(X_honest.iloc[te_r])[:, 1]",
    "    scores = [precision_at_k(proba_r, y[te_r], k) for k in K_VALUES]",
    "    random_results[name] = scores",
    "    print(f\"{name:<20} \" + \" \".join(f\"{s:>7.3f}\" for s in scores))",
    "",
    "# --- gap ---",
    "print(\"\\n=== GAP (random − grouped) — memorization signal ===\")",
    "print(f\"{'model':<20} \" + \" \".join(f\"@K={k:>3}\" for k in K_VALUES))",
    "print(\"-\" * 60)",
    "for name in [\"logistic_regression\", \"random_forest\"]:",
    "    gap = [random_results[name][i] - grouped_results[name][i] for i in range(len(K_VALUES))]",
    "    print(f\"{name:<20} \" + \" \".join(f\"{g:>+7.3f}\" for g in gap))",
    "",
    "print(\"\\nInterpretation: a positive gap means the random split was inflating scores via client memorization.\")",
    "print(\"The grouped number is the honest one to report.\")",
))

cells.append(md(
    "### Time split note\n",
    "\n",
    "This teaching slice is a **single 90-day snapshot** (cross-sectional), not a time series, so a true train-past/test-future split isn't available here. In the warehouse release (weeks 3+), `fact_content_daily_performance` is partitioned by month, enabling a genuine time split — the only split that mimics deployment for anything trend-like. That limitation is disclosed, not hidden.\n"
))