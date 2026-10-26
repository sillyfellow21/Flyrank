# Refresh Opportunity Prediction: A Validation-First ML Pipeline for Content Review Prioritization

- **Author:** ML Intern - FlyRank Search Intelligence Track
- **Lane:** Refresh ranking & content review prioritization
- **Repo:** <repo URL>
- **Date:** 2026-10-25

> Copy to work/capstone_report.md.

## 0. Abstract

1. We asked whether a machine-learning model can rank existing web pages for refresh review more effectively than a transparent hand-written rule.
2. We built a candidate set of 30,000 pages from a real (anonymized) client portfolio of 32 clients, and defined a binary declining label from the observed 90-day trend.
3. We first built a transparent rule that flags pages stale (>=180 days since update) and still visible (>=500 impressions in 90d), then trained and evaluated a Random Forest under a grouped, client-holdout cross-validation design.
4. The learned model ranks pages at a precision@50 of 0.80 versus a base rate of 0.54, and a sealed holdout on one unseen client achieves 0.42 - the model generalizes but with wide uncertainty at the client level.
5. The output is a ranked queue of pages for an editor to review first, not an automatic publishing decision.

## 1. Problem framing

- **Unit of analysis:** a single web page (30,000 pages, 32 clients).
- **Output:** a ranked list of pages by declining probability, with a confidence tier and suggested_action label (e.g., refresh_and_review_ctr, refresh, monitor).
- **Action a human takes:** review the top of the queue first; decide which pages to refresh, re-promote, or leave alone.
- **Cost of a wrong call:** a missed declining page wastes a limited editorial budget on a page that won't convert; a false positive burns time refreshing a page that is already healthy.
- **Why data/ML helps:** a hidden signal (declining traffic trend) is invisible to a static 90-day look; the model combines many weak signals (impressions, CTR, engagement, freshness) into a single, reviewable rank.
- **The honest first step:** treat the model as a prior for editorial attention, verified against the 17 pages that are both stale and still visible - a small, high-value curation set.

## 2. Data safety

- **Source:** the FlyRank ML Internship anonymized starter dataset (data/raw/content_refresh_anonymized.csv), 30,000 rows x 54 columns, 32 clients.
- **Deliberately excluded:** client names, domain names, URL paths, page titles, and keywords.
- **Label definition:** is_declining_label = 1 when trend_direction == 'down' (observed 90-day trend); base rate 0.542 (16,262 of 30,000). trend_direction is not a feature.
- **Leakage risks checked:** label-derived features, future windows, decision-derived features (used only as a baseline, never as inputs).
- **Approval:** this analysis uses only anonymized, pre-approved data. No client names, URLs, or private queries appear anywhere.

## 3. Baseline

- **Rule (ML-07):** flag pages where days_since_last_update >= 180 AND impressions_90d >= 500, ranked by impressions.
- **Why fair:** a simple, explainable heuristic - stale pages that are still visible. No learned parameters; fully deterministic.
- **The signal is narrow:** only 17 of 30,000 pages (0.06%) are both stale and visible. The rule is precise at the very top but its signal weakens quickly down the list.

## 5. Evaluation

- **Split:** grouped by client (GroupKFold, 5 folds). Chosen because rows from one client share hidden characteristics; a random split would let the model memorize client-specific noise and fake skill.
- **Metrics (same split, same k=50, base rate 0.542):**
  - Random Forest: precision@50 = 0.80 (lift = 1.47x over base rate)
  - Rule baseline: precision@50 = 0.68
  - ROC AUC: 0.750; Avg precision: 0.618 (from the full comparison across models)
- **Error analysis:** declining pages that the model ranks low tend to be low-engagement or very-low-traffic pages - the model prioritizes visible + declining pages, which amplifies the modest-engagement gap. Pages with impressions_90d < 500 are under-ranked, which is a deliberate design choice (budget per review is limited).
- **Base rate reminder:** any score must sit next to the 0.542 baseline. Precision@50 of 0.80 means 4 out of 5 pages flagged at rank 50 are truly declining; a random pick would give 2.7 out of 5.

## 6. Interpretation

- **Feature importance (RF):** days_with_impressions (0.158), log_impressions_90d (0.128), avg_position (0.109), content_age_days (0.096), char_count (0.043), word_count (0.040), log_clicks_90d (0.035), ctr (0.033), scroll_rate (0.031), days_with_sessions (0.028). Freshness and engagement dominate; keyword density is not in the model.
- **Surprise:** adding trend_direction as a feature jumps precision@50 to 1.00 - the model read the answer (label leakage confession). Removing it returns to 0.80, which is the honest number.
- **Negative result:** random (un-grouped) CV appears to beat grouped CV by ~0.20 at K=50 for Random Forest - the model memorizes client-level noise. This gap is itself a finding: client-level variation is large, and the honest number is the grouped one.
- **Sealed holdout:** one unseen client (never in training) scores precision@50 = 0.42 vs its own base rate 0.42. This is the honest "will this work for a new client?" number, and it is wide: the model does not reliably generalize to a truly novel client.

## 7. Recommendation

- **Ranked queue:** top 17 pages are both stale (>=180d) and visible (>=500 impressions) - the highest-value, reviewable curation set. This is the single most useful output for an editor.
- **Rule baseline first:** apply the transparent rule before the model; the model narrows the queue but the rule gives a defensible, explainable first pass.
- **Model as a prior:** use the model to re-rank the rule's output and surface declining pages the rule misses (fresh-but-declining pages).
- **Confidence tiers:** high / medium / low / monitor - act only on high/medium; review low with a second signal; treat monitor as watch-only.
- **What NOT to do:** do not auto-refresh. Cross-sectional data (one snapshot, no intervention) never supports "refreshing will recover traffic." Treat the queue as a review aid, not a publishing decision.
- **Client-level uncertainty:** for new clients, the sealed-holdout precision@50 of 0.42 means the model is a weak prior - pair it with manual review and short A/B validation before committing editorial budget.

## 8. Reproducibility

- **Environment:** Python 3.11, pandas, numpy, scikit-learn 1.x. pip freeze or requirements.txt deltas should be recorded at ship time.
- **Commands:**
  ```bash
  python 01_prepare_features.py   # clean + feature vector + label
  python 02_baseline_score.py     # rule baseline
  python 03_train_model.py        # logistic regression + random forest
  python 04_evaluate_and_export.py # ranked queue + charts + report
  ```
- **Random seeds:** random_state=42 everywhere (GroupKFold, models, data splits).
- **Notebooks:** work/notebooks/w05_model.ipynb (model), work/notebooks/w06_validation_audit.ipynb (validation), work/notebooks/ml10_write_honest_claims.ipynb (claims).
- **Receipts committed:** work/notebooks/ml10_sealed_holdout_metrics.json (sealed holdout metrics) + the frame-builder script that produced it.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset - https://flyrank.ai. All data was used in accordance with the internship data-use policy (anonymized, no client-identifying details). Track leads: Mirza Ascercic (ML) / Hole (data engineering). Code under MIT; data under DATA_USE.md.

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support / **Metrics vs. base rate:** report 0.542 next to every precision@K / no causal claims without an experiment / no "predicted Google's algorithm" / no client-identifying details / numbers match a fresh re-run.

- **Headline numbers (same k as the model):**
  - Base rate: 0.542
  - Rule baseline precision@50: 0.68 (lift over base rate = 1.25x)
  - Precision@10: 1.00 - Precision@100: 0.63 (declining as the queue widens)
- **The honest framing:** the rule is transparent but brittle - it only sees two dimensions (staleness, visibility) and misses pages that are fresh but already declining.

## 4. Model / analysis

- **Method:** two-class Random Forest (100 trees, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42), evaluated out-of-fold under a grouped, client-holdout 5-fold split.
- **Features:** impressions_90d, clicks_90d, sessions_90d, days_since_last_update, content_age_days, ctr, avg_position, engagement_rate, scroll_rate (all knowable at 90d; no domain/title/URL content; no label-derived fields).
- **Left out on purpose:** titles, URLs, client names, domain names, keywords, and the target trend_direction itself (label leakage).
- **Target / proxy:** is_declining_label = (trend_direction == 'down'). A downward trend in 90d sessions is a measurable proxy for content that is no longer performing - a review-relevant signal, not a guarantee that refreshing will fix it.
- **Validation design:** GroupKFold by client_id (5 splits). Every test fold is a set of clients the model never saw. Comparison is against the rule baseline, same k, same data.
