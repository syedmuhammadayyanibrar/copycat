# Synthetic Dataset Generative Process Specification

## 1. Important Framing & Methodological Disclaimer

This dataset and generator validate a **measurement methodology** using synthetic data with known ground truth. It does not and cannot prove that Grocalo's actual product (or any specific commercial system) has a novelty gap. 

The synthetic phase demonstrates that the evaluation harness can detect a novelty gap when one exists by construction. The real-data proxy (Phase 4), if evaluated, measures an ungrounded cluster-distance proxy that is structurally different from this clean synthetic decomposition.

---

## 2. Latent Factors & Creator Representation

Each creator $i \in \{1, \dots, 25\}$ possesses a latent taste preference vector $u_i \in \mathbb{R}^k$ with $k=5$.
The five underlying latent dimensions represent intrinsic creative qualities:
1. `deadpan_delivery`
2. `observational_depth`
3. `technical_rigor`
4. `absurdist_humor`
5. `visual_craft`

Vectors $u_i$ are sampled independently from a standard multivariate Gaussian and normalized to unit Euclidean length:
$$u_i \sim \mathcal{N}(0, I_k), \quad \hat{u}_i = \frac{u_i}{\|u_i\|_2}$$

---

## 3. Video Generation & Observable Features

Each video $j$ is characterized by:
- **Latent Attribute Vector** $v_j \in \mathbb{R}^k$:
  $$v_j \sim \mathcal{N}(0, I_k), \quad \hat{v}_j = \frac{v_j}{\|v_j\|_2}$$
- **Observable Categorical Features**:
  - `format`: one of `[talking_head, tutorial, screen_breakdown, sketch, field_vlog, mini_doc, silent_essay]`
  - `topic`: one of `[tech, philosophy, productivity, comedy, science, culture]`
  - `thumbnail_style`: one of `[face_closeup, minimalist_text, high_contrast_action, diagram_schematic, meme_edit]`
- **Observable Continuous Features**:
  - `duration_sec`: log-normal distribution centered around 180s to 600s
  - `cuts_per_min`: gamma distribution representing editing speed
  - `audio_energy`: beta distribution $[0, 1]$
  - `visual_contrast`: beta distribution $[0, 1]$

---

## 4. Historical Confounding & Dataset Splits

To simulate real creator history where creators repeat a narrow set of familiar production formats, each creator is assigned:
- 2 primary training formats (e.g. `talking_head`, `tutorial`)
- 1 primary thumbnail style

### Splits per creator:
1. **`train` (60 videos)**:
   - Generated exclusively using the creator's primary formats and thumbnail styles.
   - Latent factors $v_j$ vary naturally, meaning some training videos align with $u_i$ (hits) and others do not (flops).
   - This creates a **shortcut correlation**: high performance is confounded with the primary format.
2. **`held_out_similar` (25 videos)**:
   - Drawn from the same primary formats and thumbnail styles as training.
   - Tests interpolation and in-distribution generalization.
3. **`held_out_novel` (25 videos)**:
   - Drawn exclusively from novel formats (e.g. `mini_doc`, `silent_essay`, `sketch`) never seen in the creator's training history.
   - Latent factors $v_j$ are generated with both high and low taste match.
   - Tests whether a model relies on surface format shortcuts or recognizes underlying taste alignment.

---

## 5. Performance Generative Function

The observed performance $Y_{ij}$ (proxy for views / composite engagement score) follows a log-normal distribution:

$$\log(Y_{ij}) = \alpha_i + \beta_T \cdot \text{TasteMatch}_{ij} + \beta_P \cdot \text{PlatformTrend}_j + \beta_N \cdot (\text{NoveltyDist}_{ij} \times \text{TasteMatch}_{ij}) + \epsilon_{ij}$$

Where:
- **$\alpha_i \sim \mathcal{N}(10.0, 0.5^2)$**: Creator baseline scale (subscriber base / reach).
- **$\text{TasteMatch}_{ij} = \hat{u}_i^\top \hat{v}_j \in [-1, 1]$**: Intrinsic alignment between creator taste and video attributes.
- **$\beta_T = 1.8$**: Taste effect weight (major driver of audience satisfaction).
- **$\text{PlatformTrend}_j = \text{TopicBias}(\text{topic}_j) + \xi_j$**: Topic interest offset plus random platform reach shock $\xi_j \sim \mathcal{N}(0, 0.2^2)$.
- **$\beta_P = 0.8$**: Platform weight.
- **$\text{NoveltyDist}_{ij}$**: Jaccard / Gower distance in observable space between video $j$ and the training format distribution of creator $i$. (0 for familiar formats, $\approx 1.0$ for novel formats).
- **$\beta_N = 0.5$**: Novelty interaction weight. A novel video with positive taste match receives an audience discovery bonus; a novel video with negative taste match suffers audience rejection.
- **$\epsilon_{ij} \sim \mathcal{N}(0, 0.35^2)$**: Unobserved irreducible observation noise.

Performance in natural units:
$$Y_{ij} = \exp\left( \log(Y_{ij}) \right)$$

---

## 6. Partitioning: Development vs Held-Out Creators

- **Development Creators (20 creators, IDs `creator_00` to `creator_19`)**:
  Used for model development, hyperparameter selection, and validation.
- **Held-Out Creators (5 creators, IDs `creator_20` to `creator_24`)**:
  Strictly quarantined in `data/held_out/`. Untouched until the final evaluation run. No model design iterations are performed on these creators.
