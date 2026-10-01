# Copycat: Creator Taste vs. Format Memorization Evaluation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ## IMPORTANT METHODOLOGICAL FRAMING & DISCLAIMER
> This project validates an **evaluation and measurement methodology** using synthetic data with known ground truth. It **does not and cannot prove that Grocalo's actual commercial product has a novelty gap**. 
> 
> The synthetic benchmark demonstrates that this measurement protocol can detect a novelty gap when one exists by construction. The optional real-data phase (Phase 4) relies on an ungrounded cluster-distance heuristic that measures something structurally different from the clean synthetic latent decomposition. At no point should any result from this repository be overstated as empirical evidence regarding Grocalo's proprietary system or internal algorithms.

---

## 1. Research Question

> **Can a performance-prediction model trained on a creator's historical content distinguish genuine taste-based prediction from memorization of past hits, and if so, how large is the resulting "novelty gap" when evaluated on content that is taste-consistent but structurally novel?**

Creator AI tools frequently claim to "bottle up creator taste" and replicate virality. However, historical creator catalogs exhibit strong confounding: creators often repeat a narrow set of familiar production formats. Standard predictive models risk latching onto surface format templates (shortcut learning) rather than recovering the underlying creative taste dimensions.

---

## 2. Experimental Architecture

```
                                      +--------------------------+
                                      | 25 Synthetic Creators    |
                                      | 5 Latent Taste Factors   |
                                      +--------------------------+
                                                   |
                       +---------------------------+---------------------------+
                       |                                                       |
                       v                                                       v
         [20 Development Creators]                               [5 Quarantined Held-Out Creators]
         (Model design & tuning)                                 (Evaluated strictly ONCE)
                       |                                                       |
         +-------------+-------------+                                         |
         |             |             |                                         |
         v             v             v                                         v
   +-----------+ +-----------+ +-----------+                             +-----------+
   |   Train   | |  Similar  | |   Novel   |                             | Final Run |
   | (Familiar)| | (Familiar)| |  (Novel   |                             | Evaluation|
   |  Formats  | |  Formats) | |  Formats) |                             +-----------+
   +-----------+ +-----------+ +-----------+                                   |
         |             |             |                                         v
         +-------------+-------------+                             +-----------------------+
                       |                                           | Final Report          |
                       v                                           | (EVAL_REPORT.md)      |
         +---------------------------+                             +-----------------------+
         | 3 Models Behind Interface |
         | 1. Memorization Baseline  |
         | 2. Latent Taste Model     |
         | 3. Oracle Ceiling Bound   |
         +---------------------------+
                       |
                       v
         +---------------------------+
         | Post-Hoc Hungarian Match  |
         | Latent Factor Recovery    |
         +---------------------------+
```

### Phase 1: Synthetic Dataset with Known Ground Truth
- **25 Creators Total**: 20 development creators, 5 quarantined held-out creators.
- **5 Latent Factors**: `deadpan_delivery`, `observational_depth`, `technical_rigor`, `absurdist_humor`, `visual_craft`.
- **Observable Features**: Format (7 types), Thumbnail Style (5 styles), Topic (6 categories), Duration, Cuts/min, Audio Energy, Visual Contrast.
- **Splits Per Creator**:
  - `train` (60 videos): Familiar formats only (confounding surface structure with performance).
  - `held_out_similar` (25 videos): Familiar formats, varying quality.
  - `held_out_novel` (25 videos): Structurally novel formats never seen in creator history, varying quality.
- For complete equations and distribution parameters, see [DATASET_README.md](file:///c:/copycat/DATASET_README.md).

### Phase 2: Three Models Behind One Common Interface
1. **Memorization Baseline**: Nearest-neighbor RBF similarity to past high-performing videos in observable space. Predicts high performance if a video mimics the surface structure of past hits.
2. **Latent Taste Model**: Bilinear / multi-creator neural representation model learning an invariant mapping from observable features to latent dimensions $\hat{v}(x)$, paired with creator-specific taste representations $\hat{u}_i$.
3. **Oracle Model**: Direct access to ground-truth creator taste $u_i$ and video latent factors $v_j$. Defines the mathematical ceiling.
4. **Factor Identifiability**: Post-hoc Hungarian matching via `scipy.optimize.linear_sum_assignment` on the absolute correlation matrix between learned factors and true named factors.

### Phase 3: Evaluation Metrics
- **Spearman Rank Correlation ($\rho$)**: Evaluated separately on `held_out_similar` and `held_out_novel`.
- **Novelty Gap**: $\Delta_{\text{novelty}} = \bar{\rho}_{\text{similar}} - \bar{\rho}_{\text{novel}}$.
- **Memorization Bias**: $\text{Bias}_{\text{memo}} = \rho(\hat{y}, \text{SurfaceSim}) - \rho(\hat{y}, \text{TrueTaste})$.
- **Latent Factor Recovery**: Mean correlation across Hungarian-matched factor pairs.
- **Pre-Registration**: Success and Null case interpretations were pre-registered in [EVAL_NARRATIVES.md](file:///c:/copycat/EVAL_NARRATIVES.md) before executing the held-out evaluation.

### Phase 4 (Optional Stretch): Real-Data Proxy
- Public video metadata fetcher (YouTube Data API v3).
- Text/format KMeans clustering partitioning historical videos into in-cluster vs out-of-cluster (crude novelty proxy).

---

## 3. Final Held-Out Evaluation Results

The final benchmark run on the 5 quarantined held-out creators produced the following results:

| Model | Similar Spearman ($\rho_{\text{sim}}$) | Novel Spearman ($\rho_{\text{nov}}$) | Novelty Gap ($\Delta$) | Memorization Bias |
| :--- | :--- | :--- | :--- | :--- |
| **Memorization Baseline** | 0.118 $\pm$ 0.402 | **0.030** $\pm$ 0.249 | 0.088 | **+0.505** |
| **Latent Taste Model** | **0.483** $\pm$ 0.121 | **0.366** $\pm$ 0.225 | 0.118 | **-0.369** |
| **Oracle Ceiling** | 0.939 $\pm$ 0.018 | 0.954 $\pm$ 0.010 | -0.015 | -0.949 |

### Hungarian Latent Factor Recovery (Mean Absolute Correlation: `0.406`)
- `visual_craft`: **0.472**
- `absurdist_humor`: **0.480**
- `observational_depth`: **0.479**
- `technical_rigor`: **0.378**
- `deadpan_delivery`: **0.223**

### Confirmation of Pre-Registered Narrative A
The held-out results confirm **Narrative A (Success Case)**:
- The Memorization baseline collapses on novel content ($\rho_{\text{nov}} = 0.030$), exhibiting high positive memorization bias (+0.505).
- The Latent Taste Model achieves over $12\times$ higher ranking correlation on novel content ($\rho_{\text{nov}} = 0.366$), displaying negative memorization bias (-0.369) and strong factor recovery (0.406).
- Full per-creator breakdowns are detailed in [EVAL_REPORT.md](file:///c:/copycat/EVAL_REPORT.md).

---

## 4. Repository Layout

```
copycat/
├── DATASET_README.md             <- Auditable generative process documentation
├── EVAL_NARRATIVES.md            <- Pre-registered success/null evaluation narratives
├── EVAL_REPORT.md                <- Final held-out evaluation report
├── copycat/
│   ├── data/
│   │   ├── generator.py          <- Synthetic data generator (taste, confounding, splits)
│   │   ├── schema.py             <- Dataclasses and feature constants
│   │   └── real_proxy.py         <- Real YouTube API fetcher & cluster proxy
│   ├── models/
│   │   ├── base.py               <- Abstract model base class
│   │   ├── memorization.py       <- Surface template matching nearest-neighbor model
│   │   ├── taste.py              <- Bilinear latent taste factor model
│   │   └── oracle.py             <- Ground-truth ceiling model
│   └── evaluation/
│       ├── metrics.py            <- Spearman rho, novelty gap, memorization bias
│       ├── identifiability.py    <- Hungarian factor matching algorithm
│       └── harness.py            <- Unified evaluation harness
├── scripts/
│   ├── generate_data.py          <- Generates dev and quarantined held-out data
│   ├── run_dev_eval.py           <- Development evaluation runner & model checkpointer
│   ├── run_final_eval.py         <- Final un-tuned held-out evaluation runner
│   └── run_real_proxy.py         <- Phase 4 real data proxy runner
└── tests/
    ├── test_generator.py         <- Unit tests for generator
    ├── test_models.py            <- Unit tests for all three models
    └── test_evaluation.py        <- Unit tests for metrics and Hungarian matching
```

---

## 5. Execution & Verification

### Running Unit Tests
```bash
python -m unittest discover tests
```

### Regenerating Synthetic Datasets
```bash
python scripts/generate_data.py
```

### Running Development Evaluation
```bash
python scripts/run_dev_eval.py
```

### Running Final Held-Out Evaluation
```bash
python scripts/run_final_eval.py
```

### Running Phase 4 Real-Data Proxy
```bash
python scripts/run_real_proxy.py
```
