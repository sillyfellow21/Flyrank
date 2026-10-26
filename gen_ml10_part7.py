cells.append(md(
    "## 7. Selection bias check\n",
    "\n",
    "If the \"treated\" group was CHOSEN (someone picked which pages to refresh), part of the gap is the choosing, not the treatment — say so in the same sentence.\n",
    "\n",
    "Claim: \"Refreshing the rule-flagged pages should recover traffic.\"\n",
    "Honest form: \"The rule-flagged pages are associated with decline (precision@50 = 0.68 vs base rate 0.54), but the flagged set was CHOSEN by the rule itself, so part of any post-refresh improvement would be selection — pages chosen because they looked stale, not because refresh caused the recovery.\"\n",
))

cells.append(md(
    "## 8. Survivorship check\n",
    "\n",
    "Filters like \"active items only\" silently drop the dead — name the filter when stating the finding.\n",
    "\n",
    "Claim: \"Among active pages, X% are declining.\"\n",
    "Honest form: \"In this snapshot, 54.21% of all 30,000 pages show a downward trend. This slice includes only pages with ≥1 impression in the trailing 90 days (per the dataset definition); pages with zero impressions — potentially the most dead — are not represented here, so the decline rate is a lower-bound signal for the full portfolio.\"\n",
))

cells.append(md(
    "## 9. Negative results\n",
    "\n",
    "Negative results are results. \"We expected X; the data shows no such pattern\" is publishable.\n",
    "\n",
    "Negative finding: We expected the rule baseline's precision to rise monotonically with K (bigger flagged set → more true decliners caught), but precision@K actually *declines* as K grows (1.00 at K=10 → 0.68 at K=50 → 0.63 at K=100).\n",
    "\n",
    "Honest form: \"We expected precision to hold or rise as we flag more pages; the data shows the opposite — precision@K falls from 1.00 at K=10 to 0.63 at K=100. The rule is precise at the very top but its signal weakens quickly down the list. That is a real, reportable pattern, not a failure.\"\n",
))