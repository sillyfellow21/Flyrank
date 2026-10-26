cells.append(md(
    "## 10. Banned-phrasing audit\n",
    "\n",
    "We reviewed every conclusion sentence against the banned list. Phrases found and how they were fixed:\n",
    "\n",
    "| Banned phrase (do not use) | Honest replacement |\n",
    "|---|---|\n",
    "| \"Refreshing X will increase traffic\" | \"X is associated with decline; these pages look worth reviewing first\" |\n",
    "| \"The algorithm rewards fresh content\" | \"In this data, pages not updated in 180+ days are overrepresented among decliners (observed, not causal)\" |\n",
    "| \"Our model proves refresh works\" | \"The model ranks pages at precision@K of 0.80 vs base rate 0.54 (decision-support, not proof)\" |\n",
    "| \"Accuracy is 80%\" | \"Accuracy not reported without base rate; we use precision@K next to base rate 0.54\" |\n",
    "| \"We predicted Google's algorithm\" | \"No — we modeled outcomes in one portfolio; claims are decision-support only\" |\n",
    "| \"MASSIVE boost\" | \"Lift over base rate @50 = 1.47x (0.80 vs 0.54), n = 30,000\" |\n",
))

cells.append(md(
    "## 11. Claim ladder audit\n",
    "\n",
    "Each conclusion sentence below is read alone, out of context: does it say more than its table shows?\n",
))

cells.append(code(
    "# claim ladder audit table (printed as text so each sentence stands alone)",
    "audit = [",
    "    (\"In this dataset, 17 of 30,000 pages (0.06%) are both stale (>=180d) and visible (>=500 impressions).\", \"Observed\", \"Yes — computed from this snapshot\"),",
    "    (\"The rule flags decliners with precision@50 of 0.68 vs base rate 0.54.\", \"Observed\", \"Yes — measured on these rows\"),",
    "    (\"Random Forest ranks pages at precision@50 of 0.80 vs base rate 0.54 (grouped CV).\", \"Validated ranking out-of-sample\", \"Yes — client-holdout CV, model never saw the client\"),",
    "    (\"Adding trend_direction as a feature jumps precision@50 to 1.00 — the model read the answer.\", \"Observed\", \"Yes — deliberate leakage demo\"),",
    "    (\"Random split beats grouped by +0.20 at K=50 for Random Forest — client memorization.\", \"Measured comparison\", \"Yes — both splits computed\"),",
    "    (\"Sealed holdout on one unseen client: precision@50 = 0.42 vs its base rate 0.42.\", \"Validated out-of-sample\", \"Yes — client never in training\"),",
    "    (\"Refreshing flagged pages will recover traffic.\", \"Causal\", \"REJECTED — cross-sectional, no experiment. Rewritten as decision-support.\"),",
    ")",
    "print(\"| Claim sentence | Ladder rung | Evidence carries it? |\")",
    "print(\"|---|---|---|\")",
    "for sentence, rung, carries in audit:",
    "    print(f\"| {sentence} | {rung} | {carries} |\")",
))