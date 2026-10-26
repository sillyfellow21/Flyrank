cells.append(md(
    "## 4. Attack checklist (run before believing any score)\n",
    "\n",
    "The skill's checklist, applied to this notebook:\n",
    "\n",
    "| Check | Status | Notes |\n",
    "|---|---|---|\n",
    "| Timeline drawn — all features strictly before the label window | ✅ | Honest features are trailing-90-day aggregates; `_last_30d` excluded |\n",
    "| No label-derived or sibling columns in features | ✅ | `trend_direction`, `trend_pct`, `is_declining` excluded |\n",
    "| No product flags / existing-system scores as features | ✅ | `health_score`, `is_underperformer`, `is_quick_win`, etc. excluded |\n",
    "| Population selection checked for outcome-window information | ⚠️ | All 30k rows kept; no filtering on the label month (disclosed) |\n",
    "| Split grouped by repeating entity (client) | ✅ | GroupKFold by `client_id` |\n",
    "| Base rate printed next to every metric | ✅ | Base rate printed throughout (computed live in §1) |\n",
    "| Top feature importance sanity-checked | ✅ | §3 above — no \"too good\" feature |\n",
    "| Metrics recomputed out-of-fold, never in-sample | ✅ | OOF predictions via GroupKFold |\n",
    "| Deliberately add a leaky feature → score jumps to ~1.0 | ✅ | §2a/2b/2c above (harness proven) |\n",
    "| Swap random split for grouped split; report both | ✅ | §5 below |\n",
    "",
))

cells.append(code(
    "# --- leaky-feature harness proof (already demonstrated in §2) ---",
    "print(\"Harness proof — deliberately added leaky features and the score moved:\")",
    "print(f\"  label-derived  : {p_honest:.4f} → {p_leaky_label:.4f}  (Δ{p_leaky_label-p_honest:+.4f})\")",
    "print(f\"  product flags  : {p_honest:.4f} → {p_leaky_flags:.4f}  (Δ{p_leaky_flags-p_honest:+.4f})\")",
    "print(f\"  overlapping win: {p_honest:.4f} → {p_leaky_window:.4f}  (Δ{p_leaky_window-p_honest:+.4f})\")",
    "print(\"\\nEach jump confirms the harness detects leakage. Leaks removed; honest number kept.\")",
))