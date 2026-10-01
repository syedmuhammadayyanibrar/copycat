# Pre-Registered Evaluation Narratives

## 1. Methodological Purpose & Framing

This evaluation document sets pre-registered interpretation criteria for the copycat benchmark before executing the final evaluation run on the quarantined held-out creators (`creator_20` through `creator_24`).

**Critical Framing**:
This project benchmarks a **measurement methodology** on synthetic data where ground-truth taste and performance mechanics are known by construction. A failure of the memorization baseline or a success of the taste model validates that this evaluation protocol can detect a novelty gap when present. It does **not** evaluate or prove any claim regarding Grocalo's commercial system or any real-world creator tools.

---

## 2. Quantitative Decision Boundaries

The benchmark computes the Novelty Gap:
$$\Delta_{\text{novelty}} = \bar{\rho}_{\text{similar}} - \bar{\rho}_{\text{novel}}$$
where $\bar{\rho}$ is the mean creator-level Spearman rank correlation between predicted and true video performance.

We also compute the Memorization Bias from diagnostic correlations:
$$\text{Bias}_{\text{memorization}} = \rho(\hat{y}, \text{SurfaceSim}) - \rho(\hat{y}, \text{TrueTaste})$$

---

## 3. Narrative A: Success Case (Methodology Validated)

### Trigger Criteria:
1. The **Memorization Baseline** exhibits a pronounced Novelty Gap ($\Delta_{\text{novelty}} \ge 0.25$), showing strong degradation when transitioning from familiar formats to novel formats.
2. The **Taste Model** maintains rank correlation on novel content, achieving:
   - Novel Spearman correlation $\bar{\rho}_{\text{novel}} > \bar{\rho}_{\text{novel}}^{(\text{Memorization})} + 0.15$
   - A significantly compressed Novelty Gap ($\Delta_{\text{novelty}} \le 0.15$).
3. The **Oracle Model** defines the theoretical ceiling with near-zero Novelty Gap ($\Delta_{\text{novelty}} \le 0.05$) and high correlation across both splits ($\bar{\rho} \ge 0.70$).
4. Hungarian factor recovery identifies corresponding latent factors with mean recovery correlation $r_{\text{mean}} \ge 0.40$.

### Narrative Interpretation:
The experimental results demonstrate that standard surface-similarity memorization collapses when confronted with structurally novel formats, despite constant underlying creative taste. The Latent Taste Model successfully disconfounds surface production formats from intrinsic taste preferences, generalizing its performance predictions to novel creative mediums. This proves the validity and sensitivity of the copycat evaluation protocol for detecting shortcut learning in creator-content AI models.

---

## 4. Narrative B: Null Case (Methodological Limitation)

### Trigger Criteria:
1. The **Taste Model** fails to significantly outperform the Memorization Baseline on novel content ($\bar{\rho}_{\text{novel}}^{(\text{Taste})} - \bar{\rho}_{\text{novel}}^{(\text{Memorization})} < 0.10$).
2. OR both models suffer similar performance collapses on the novel split.
3. OR Hungarian factor recovery correlation remains near random chance ($r_{\text{mean}} < 0.25$).

### Narrative Interpretation:
The evaluation harness did not establish a distinct advantage for the latent taste architecture over surface memorization on novel content. 

**Attribution of Null Result**:
This outcome reflects a limitation in either:
- The synthetic generative process (insufficient signal-to-noise ratio or excessive entanglement between surface features and latent factors), OR
- The specific architecture / optimization of the latent taste representation model (e.g. inability to isolate orthogonal taste dimensions from small-sample creator histories).

Under no circumstances should this null result be interpreted as empirical evidence about the performance or internal workings of Grocalo's real-world product.
