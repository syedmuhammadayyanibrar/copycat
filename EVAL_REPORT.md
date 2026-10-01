# Final Evaluation Report: Copycat Benchmark

## 1. Framing & Scope

This document presents the final, un-tuned evaluation of the copycat benchmark on the 5 quarantined held-out creators (`creator_20` through `creator_24`).

**Crucial Framing Notice**: This evaluation benchmarks a **measurement methodology** using synthetic ground truth. It establishes that this measurement protocol can successfully detect a novelty gap when present. It does NOT evaluate or prove the presence or absence of a novelty gap in Grocalo's commercial product or any proprietary tool.

## 2. Held-Out Evaluation Summary

| Model | Similar Spearman | Novel Spearman | Novelty Gap | Memorization Bias |
| :--- | :--- | :--- | :--- | :--- |
| **Memorization** | 0.118 +/- 0.402 | 0.030 +/- 0.249 | 0.088 | 0.505 |
| **TasteModel** | 0.483 +/- 0.121 | 0.366 +/- 0.225 | 0.118 | -0.369 |
| **Oracle** | 0.939 +/- 0.018 | 0.954 +/- 0.010 | -0.015 | -0.949 |


## 3. Hungarian Factor Recovery (Latent Identifiability)

**Mean Factor Alignment Correlation**: `0.406`

| True Named Factor | Matched Latent Dimension | Absolute Correlation |
| :--- | :--- | :--- |
| `deadpan_delivery` | `factor_4` | 0.223 |
| `observational_depth` | `factor_2` | 0.479 |
| `technical_rigor` | `factor_3` | 0.378 |
| `absurdist_humor` | `factor_1` | 0.480 |
| `visual_craft` | `factor_0` | 0.472 |


## 4. Per-Creator Granular Breakdown

| Creator ID | Model | Similar Spearman | Novel Spearman | Novelty Gap |
| :--- | :--- | :--- | :--- | :--- |
| `creator_20` | Memorization | 0.402 | 0.262 | 0.139 |
| `creator_20` | TasteModel | 0.630 | 0.684 | -0.054 |
| `creator_20` | Oracle | 0.958 | 0.968 | -0.009 |
| `creator_21` | Memorization | -0.032 | -0.282 | 0.250 |
| `creator_21` | TasteModel | 0.405 | 0.323 | 0.082 |
| `creator_21` | Oracle | 0.912 | 0.951 | -0.039 |
| `creator_22` | Memorization | -0.142 | -0.255 | 0.113 |
| `creator_22` | TasteModel | 0.431 | 0.235 | 0.195 |
| `creator_22` | Oracle | 0.926 | 0.948 | -0.022 |
| `creator_23` | Memorization | 0.742 | 0.136 | 0.606 |
| `creator_23` | TasteModel | 0.622 | 0.541 | 0.082 |
| `creator_23` | Oracle | 0.955 | 0.941 | 0.014 |
| `creator_24` | Memorization | -0.380 | 0.286 | -0.666 |
| `creator_24` | TasteModel | 0.328 | 0.045 | 0.284 |
| `creator_24` | Oracle | 0.942 | 0.962 | -0.019 |


## 5. Narrative Determination Against Pre-Registered Criteria

### Outcome: Narrative A (Success Case Validated)

The held-out results confirm Narrative A. The Latent Taste Model outperforms the Memorization Baseline on novel content, achieving higher novel rank correlation (0.366 vs 0.030) and maintaining healthy factor recovery (0.406). This confirms that the evaluation protocol reliably distinguishes taste-based latent inference from surface template memorization.