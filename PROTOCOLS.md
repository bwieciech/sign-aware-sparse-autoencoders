Below is a **formal design document** you can hand to junior researchers to implement three protocols that demonstrate, quantify, and apply our **Sign‑aware Gated Sparse Autoencoder (SG‑Gated SAE)**. I’ve written it as if I’m onboarding a new applied science team and will call out pitfalls, checklists, and **acceptance criteria** at each step.

---

## 0) Executive summary

We will execute three protocols:

1. **Toy 2‑D Organization & Collapse** – a reproducible 2‑D synthetic experiment (à la *Toy Models of Superposition*) that shows how correlated and anti‑correlated feature sets are arranged by different autoencoders and that **anti‑correlated features collapse into a single latent** with the **right/left sign** in our SG‑Gated SAE. This mirrors the “organization of correlated and anticorrelated features” visualization in the toy‑models paper and adapts it to our architecture.&#x20;

2. **Capacity, parameter, and compute audit** – a controlled study (synthetic + one real LM site) that **quantifies “wasted capacity”** and the **net parameter/compute trade‑off**. We will (i) measure how many dictionary elements each method needs to reach target fidelity, (ii) compute the exact parameter delta per anti‑correlated pair collapsed by SG‑Gated SAE, and (iii) measure training/Inference wall‑clock deltas versus baselines. The Gated‑SAE training recipe (with the auxiliary frozen‑decoder loss and weight‑tying) is our baseline reference for training cost. &#x20;

3. **Anomaly detection with SAEs** – train on “normal” activations and detect distribution shift / injected anomalies using **reconstruction error + latent statistics (including sign patterns)**. This tests a practical application and probes whether **signed latents** provide cleaner detection signals than non‑signed SAEs.

All three protocols share core infrastructure (data pipeline, training harnesses, metrics, logging, plots) to minimize work.

---

## 1) Background, terminology, and code you’ll use

### Architectures (what we compare)

* **Baseline SAE (ReLU)** – single linear encoder with ReLU and linear decoder trained with L2 reconstruction + L1 sparsity on latents; decoder columns are unit‑norm. (This is the “baseline SAE” used in the literature.)&#x20;
* **Gated SAE (sign‑agnostic)** – separates **feature detection** (gating) from **magnitude estimation**; the L1 penalty is applied to the gate preactivations; training uses an **auxiliary frozen‑decoder** loss; weight‑tying gives a **Jump‑ReLU** view at inference. &#x20;
* **SG‑Gated SAE (ours)** – extends the Gated SAE so each latent can fire **with a sign** (positive/negative), allowing a single latent to represent **two anti‑correlated features** that lie on (nearly) the same axis. (See your `GatedSAE(sign_aware=True)` code: separate thresholds for right/left, and separate rescale parameters for pos/neg.)

> Implementation note: Our training objective mirrors the Gated‑SAE objective (reconstruction + L1 on gate preactivations + auxiliary “via‑gate” reconstruction with **frozen decoder**). That auxiliary term requires **running the decoder twice** per step (main + frozen), which increases training compute versus a classic SAE (the Gated SAE paper reports **≤1.5×** training compute for the gated loss), and we inherit that.&#x20;

### Metrics (what we will report)

* **L0 (sparsity)** – mean number of active latents.
* **Loss recovered** – splice reconstructions back into an LM and measure cross‑entropy relative to identity vs. zero ablation; 100% means perfect reconstructions, 0% equals zero‑ablation.&#x20;
* **Relative reconstruction bias γ** (shrinkage diagnostic) – γ≈1 means unbiased; γ<1 indicates shrinkage. (We’ll use it when appropriate.)&#x20;
* **Alignment/angle analyses** – angles between learned decoder directions and ground‑truth features (toy) or between methods (LM site).
* **Collapse rate (anti‑corr.)** – fraction of anti‑correlated feature pairs captured by a **single** signed latent (SG‑Gated) vs. **two** unsigned latents (others).

---

## 2) Protocol A: Toy 2‑D Organization & Collapse

**Goal.** Reproduce and extend the classic 2‑D visualization for correlated/anti‑correlated features to demonstrate that **anti‑correlated pairs occupy one latent with signed activations** in SG‑Gated SAE, while baselines need two opposite directions. (The original organization experiment is described in the “Organization of Correlated and Anticorrelated Features” section; we will mirror that setup and plots.)&#x20;

### A.1 Experimental design

1. **Data generation (2D)**

   * Latent “true” features live on unit circle in **m=2**. Draw **K bundles**:

     * **Correlated bundles**: all features in the bundle either co‑occur or are all zero. (As in the paper’s setup.)&#x20;
     * **Anti‑correlated bundles**: at most **one** feature in the bundle is non‑zero at a time.&#x20;
   * Sample magnitude ∼ Uniform\[0,1] when on; probability of being **off** shared within a correlated bundle and mutually exclusive within an anti‑correlated bundle (exactly as specified in the paper’s definitions).&#x20;
   * Linearly mix these features to 2D observations.

2. **Models to train**

   * **Baseline SAE (ReLU)** with unit‑norm decoder columns.
   * **Gated SAE (sign‑agnostic)** (paper settings: L2 + L1 on gate preactivations + auxiliary via‑gate loss with frozen decoder).&#x20;
   * **SG‑Gated SAE (ours)** with `sign_aware=True` (right/left offsets, pos/neg magnitude paths).

3. **Training recipe (shared unless stated)**

   * Optimizer: Adam(β1=0.0, β2=0.999). Batch size 4–8k; 200–300k steps are sufficient. (Matches the Gated‑SAE training practice.)&#x20;
   * Decoder columns constrained to unit norm.
   * **Resampling** of dead latents as in your `resample_dead_latents` stub and the Gated‑SAE practice. (Resampling periodically re‑initializes persistently inactive features.)&#x20;
   * For Gated/SG‑Gated, **auxiliary “via‑gate” reconstruction uses a *frozen* decoder** (stop‑gradient) to ensure the L1 only shapes feature detection.&#x20;

4. **Visualizations**

   * Standardize solutions by rotating/flipping to a canonical orientation (principal axis alignment), then **plot decoder columns on the unit circle**; color points by bundle and by bundle type (correlated vs anti‑correlated). (Same presentation used in the toy‑models paper; see the figure on their correlated/anti‑correlated page.)&#x20;
   * For SG‑Gated, **overlay sign usage**: for each latent, measure **Pr(latent>0)** and **Pr(latent<0)** when the corresponding bundle fires; depict with a radial “+”/“–” tick mark on the circle.

5. **Measurements**

   * Reconstruction MSE; **L0**; decoder/ground‑truth alignment angles; **collapse rate** = fraction of anti‑correlated pairs represented by one signed latent (SG‑Gated) vs two opposite decoders (others).

### A.2 Implementation checklist & notes

* **Data seed control** – fix seeds for bundle geometry and activation draws; store seed in artifact metadata.
* **Width setting** – choose width ≥ total #features so baselines *can* learn 1:1 if they want (we want to observe *preference*, not force).
* **Optimization trap** – the Gated‑SAE auxiliary loss must **freeze** the decoder; leaving it unfrozen degrades the intended separation (ablation confirms this).&#x20;
* **Compute tip** – in SG‑Gated, you can reuse **one** `pre_activation = W_enc @ x_centered` and compute
  `z_pos = exp(r_mag_pos) ⊙ pre_activation + b_mag`,
  `z_neg = exp(r_mag_neg) ⊙ pre_activation + b_mag`,
  avoiding extra encoder matmuls. (The Jump‑ReLU interpretation shows magnitude paths are scaled versions of the same directions.)&#x20;

### A.3 Acceptance criteria

* **Qualitative**: Figures clearly exhibit the canonical behavior reported in toy‑models—correlated bundles arranged roughly orthogonally; anti‑correlated features arranged in opposite directions under baselines; **SG‑Gated shows collapse (one direction with signed usage)**.&#x20;
* **Quantitative**:

  * **Collapse rate** ≥ **80%** of anti‑correlated pairs use **one** latent in SG‑Gated; ≤ **30%** in Baseline/Gated.
  * **Fidelity parity**: SG‑Gated MSE within **+5%** of the best baseline on the same data at matched L0 ±10%.
  * **Sparsity**: L0 of SG‑Gated ≤ Baseline at matched MSE (or the Pareto curve is not worse).

Deliverables: notebook + saved figures; CSV with metrics per run; README explaining seeds and config.

---

## 3) Protocol B: Capacity, parameter, and compute audit

**Goal.** Provide a **formal assessment** of capacity savings from sign‑aware latents when anti‑correlated features exist, counting **parameters**, **width**, and **compute** at train/inference time. We will do this in (i) synthetic settings with many anti‑correlated pairs and (ii) one LM site (e.g., Pythia‑2.8B MLP‑out layer) where we can measure **Loss‑Recovered vs. L0** Pareto curves.

### B.1 What we will measure

1. **Capacity (width) to reach target fidelity**

   * For each method, the **minimum width** needed to reach a fixed reconstruction target (MSE or Loss‑Recovered ≥ target).
   * Report **L0 at that point**.

2. **Parameter accounting (per latent and per collapsed pair)**

   * **Baseline SAE (ReLU)** per latent: `2·d_in + 1` (one encoder column + one decoder column + encoder bias).
   * **Gated SAE (sign‑agnostic)** adds `r_mag` and `b_mag`: `2·d_in + 3`. (Weight‑tying ensures only **\~2·M** extra params vs Baseline overall.)&#x20;
   * **SG‑Gated SAE** adds four more scalars per latent (`act_offset_right`, `act_offset_left`, `r_mag_pos`, `r_mag_neg`): `2·d_in + 7`.
   * **Net saving per anti‑correlated pair collapsed** (two unsigned latents → one signed latent):

     * vs **Baseline SAE**: `(4·d_in + 2) − (2·d_in + 7) = 2·d_in − 5`.
     * vs **Gated SAE**: `(4·d_in + 6) − (2·d_in + 7) = 2·d_in − 1`.
     * For any realistic `d_in ≥ 64`, savings are dominated by `~2·d_in` parameters **per collapsed pair** (the extra 5 or 1 scalars are negligible).

3. **Compute**

   * **Training**: Gated objectives require **two decodes** per step due to the **auxiliary frozen‑decoder loss** (≈ **≤1.5×** baseline training compute at matched width). SG‑Gated inherits this.&#x20;
   * **Encoder matmuls** can be shared: compute `pre_activation = W_enc @ x_centered` once and reuse for magnitude paths (both sign branches), so **inference encoder cost is roughly the same as Gated** (not 2–3×). (Jump‑ReLU view.)&#x20;

### B.2 Experimental design

**Synthetic capacity sweep**

* Generate `P` anti‑correlated pairs plus `Q` correlated bundles (same generator as Protocol A but in **higher `d_in`**, e.g., `d_in∈{128,512}`) to make param effects non‑trivial.
* Train each method across widths `M` on a grid; for each run record the fidelity (MSE), **L0**, and parameter count.
* Plot **target‑fidelity isoclines**: minimal width `M*` needed to beat the target MSE; report **% width reduction** and **param savings per sample** due to collapses.

**LM site capacity sweep**

* Choose one model/site used in the Gated‑SAE paper (e.g., **Pythia‑2.8B MLP‑out layer L=16**), where **Loss‑Recovered vs L0** is standard.&#x20;
* Train Baseline SAE, Gated SAE, and SG‑Gated SAE across L1 coefficients; reproduce **Pareto curves** (loss‑recovered vs L0) and compare SG‑Gated to the Gated baseline practice (equalize compute as in the paper when comparing).&#x20;
* Add a **collapse analysis**: cluster learned decoder directions that are near‑antipodal with high **negative co‑exclusion** of their latents; count how many such pairs SG‑Gated represents using **one** signed latent.

### B.3 Implementation checklist & notes

* **Unit‑norm constraint** on decoder columns every step.
* **Equalize compute** when drawing Pareto frontiers (the Gated paper increases baseline width by 1.5× to match compute; we will replicate that convention when comparing to Gated).&#x20;
* **Anti‑corr. pair detector (for capacity audits)**:

  1. Label anti‑correlated generating pairs (synthetic) OR detect in learned latents by: large **negative correlation** of co‑activation and **decoder directions ≈ antipodal**.
  2. For SG‑Gated, declare a **collapse** if one latent’s decoder is within `≤10°` of both pair directions and its activation **sign** is consistent with which feature was active.

### B.4 Acceptance criteria

* **Target‑fidelity width**: In synthetic experiments with ≥50% anti‑correlated pairs, SG‑Gated requires **≈50% fewer latents for those pairs** (one per pair) at matched MSE ±5%.
* **Parameter accounting**: Reported per‑pair savings match the formula above within numerical tolerance; a table with `(P, d_in) → params saved` is provided.
* **Compute**: Measured training step time overhead ≤ **1.6×** vs Baseline SAE at matched width (tracks the paper’s ≤1.5× guidance), with inference cost within **±10%** of Gated after encoder‑reuse optimization.&#x20;

Deliverables: two reports (synthetic, LM site) with Pareto plots, width/param tables, and an ablation showing that turning **off sign‑awareness** loses the collapse benefit.

---

## 4) Protocol C: Anomaly detection with SAEs (application study)

**Goal.** Evaluate whether SAEs can act as **unsupervised anomaly detectors** and whether **signed latents** provide **cleaner, more separable** detection signals.

### C.1 Experimental design

1. **Choose model/site**

   * Use a small‑to‑mid LLM with public weights (e.g., **Pythia‑2.8B**). Train SAEs at **MLP output** and one **residual** site where Gated SAEs are known to work and are easy to evaluate with loss‑recovered if needed.&#x20;

2. **Normal vs. anomalous data**

   * **Normal**: held‑out text from the same distribution as pretraining data (e.g., OpenWebText subset).
   * **Anomalies** (construct multiple types):

     * **Distribution shift** (e.g., code/docstrings if “normal” is prose).
     * **Perturbations**: token‑swap noise, rare‑character bursts, or adversarial n‑gram insertions; keep perturbations light (≤5–10% tokens) to make the task non‑trivial.
     * **Activation‑level noise**: add small Gaussian noise to hidden states for a subset to separate “model‑internal corruption” from text‑side shift.

3. **Train SAEs** (normal data only)

   * Baseline SAE (ReLU), Gated SAE, and SG‑Gated SAE with the same recipe and width (match compute as needed if comparing loss‑recovered).&#x20;
   * Store **training distribution summaries** of reconstruction errors and latent statistics (per‑latent means, variances, sign‑usage histograms).

4. **Detection signals** (evaluate on a balanced normal/anomaly set)

   * **Reconstruction MSE**.
   * **Latent usage shift**:

     * Total **L0 delta** (firing count shift).
     * For SG‑Gated: **sign asymmetry** – e.g., the **ratio** `E[|neg activations|]/E[|pos activations|]` for latents that show strong directional selectivity on normal data.
   * **Composite score**: a two‑feature logistic combining MSE and sign asymmetry.

5. **Evaluation**

   * ROC‑AUC, PR‑AUC per anomaly type.
   * **At equal FPR=1%**, report **TPR** (sensitivity).
   * Analyze **which signed latents** carry the signal: inspect their top activating contexts.

### C.2 Implementation checklist & notes

* **Do not leak anomalies into training** – ensure a clean split.
* **Calibrate thresholds** on *validation* normal only.
* **Sign features**: for SG‑Gated, cache per‑latent baseline sign distribution on normal. Deviations (e.g., a latent that is mostly positive on normal flipping negative) are strong signals.
* **Freezing decoder in auxiliary** is still important; keep that switch on.&#x20;
* **Compute**: use batch logging of **γ** to ensure no extreme shrinkage; we expect **γ≈1** for Gated variants.&#x20;

### C.3 Acceptance criteria

* **AUC**: SG‑Gated ≥ **0.85 ROC‑AUC** on at least two anomaly types; exceeds Baseline SAE and matches or exceeds sign‑agnostic Gated on average.
* **Low‑FPR regime**: At **1% FPR**, SG‑Gated achieves **≥10% absolute** higher TPR than Baseline SAE on at least one anomaly type involving **directional** signal (e.g., punctuation/noise bursts that induce sign flips in certain latents).
* **Interpretability**: For the top 5 anomaly‑indicative **signed** latents, qualitative inspection of top contexts yields a **plausible directional hypothesis** (e.g., “positive direction ≈ quote‑continuation; negative ≈ line‑noise punctuation”).

Deliverables: AUC plots, threshold tables, qualitative latent cards.

---

## 5) Shared engineering plan

### 5.1 Repository layout (suggested)

```
sgnsae/
  models/
    sae_base.py                 # baseline scaffold (yours)
    gated_sae.py                # sign_agnostic & sign_aware=True
    baseline_relu_sae.py        # simple SAE for Protocol B/C baseline
  train/
    train_sae.py                # common training loop
    losses.py                   # baseline & gated losses (frozen-decoder aux)
    resample.py                 # dead-latent resampling
  data/
    toy2d.py                    # Protocol A data generator
    lm_activations.py           # hooks to dump LM activations for Protocol B/C
  eval/
    metrics.py                  # L0, loss recovered hooks, γ
    capacity.py                 # width/param/compute accounting
    visualize_toy2d.py          # plots to unit circle & sign overlays
    pareto.py                   # Pareto curves
    anomaly.py                  # scoring and ROC/PR calculations
  scripts/
    run_protocol_A.sh
    run_protocol_B_synth.sh
    run_protocol_B_lm.sh
    run_protocol_C.sh
  reports/
    ...
```

### 5.2 Training details to standardize

* **Losses**

  * Baseline SAE: `L = ||x - x_hat||^2 + λ||f||_1`, unit‑norm decoder columns.
  * (SG‑)Gated SAE: `L = ||x - x̂(f̃)||^2 + λ ||ReLU(π_gate)||_1 + ||x - x̂_frozen(ReLU(π_gate))||^2`. (Frozen decoder for the auxiliary term; reference pseudo‑code available in the Gated‑SAE paper.)&#x20;
* **Resampling** – use your `resample_dead_latents` function; threshold tuned on a small pilot; resume with LR warm‑up after resampling (mirrors practice).&#x20;
* **Normalization** – decoder columns renormalized each step.

### 5.3 Compute & optimization notes

* **Matmul reuse for SG‑Gated** – compute `pre_activation = W_enc @ (x - b_dec)` once and synthesize both `z_pos` and `z_neg` with vectorwise rescaling and bias (saves encoder FLOPs).
* **Auxiliary decode** – the second decode dominates the extra training cost (expect up to \~1.5× versus classic SAE). Keep batch size constant across methods when timing.&#x20;
* **Stability** – track **γ** during training (expect γ≈1 for Gated variants; γ<1 indicates shrinkage typical of baseline SAE).&#x20;

### 5.4 How to compute “parameters saved”

For a given run:

```
params_baseline = (2*d_in + 1) * M_baseline + d_in    # +d_in once for b_dec shared
params_gated    = params_baseline + 2*M_gated         # +r_mag + b_mag
params_sg       = params_gated + 4*M_sg               # + act_offsets + r_mag_pos/neg
```

For *each* anti‑correlated pair that collapses in SG‑Gated:

```
Δparams_vs_baseline ≈ (4*d_in + 2) - (2*d_in + 7) = 2*d_in - 5
Δparams_vs_gated    ≈ (4*d_in + 6) - (2*d_in + 7) = 2*d_in - 1
```

Include these in tables alongside **observed collapse counts**. (The weight‑tying detail that keeps Gated SAEs only **\~2·M extra scalars total** is documented; we’ll use that to explain why our added per‑latent scalars are negligible versus saved `2·d_in` vectors.)&#x20;

---

## 6) Risk log & troubleshooting

* **Decoder not frozen in aux loss** → the gate learns to reconstruct via moving the decoder, defeating the separation of detection/magnitude and invalidating the argument; check gradients stop at the frozen path. Ablation in the Gated paper shows this matters.&#x20;
* **No visible collapse in Protocol A** → increase training steps, ensure enough anti‑correlated sets, and widen slightly so the model has freedom to choose geometric arrangements.
* **Pareto front non‑monotonic** → typical SAE training artifacts; re‑run with slightly different λ or seeds; this is also discussed in Gated‑SAE appendices.&#x20;
* **Anomaly detection degenerate** → your anomalies might be too strong; dial down perturbation level to keep ROC‑AUC < 1 for all methods.

---

## 7) Reporting templates (what to hand back)

* **Protocol A**:

  * Figure 1: unit‑circle plots (Baseline, Gated, SG‑Gated).
  * Table: collapse rate, MSE, L0.
  * Short narrative tying to toy‑models organization figure.&#x20;

* **Protocol B (synthetic)**:

  * Plot: target‑fidelity width vs. % anti‑correlated pairs.
  * Table: params saved = `(#collapsed_pairs) * (2*d_in - 5)` (vs Baseline) and `(2*d_in - 1)` (vs Gated), plus measured step‑time ratios.

* **Protocol B (LM site)**:

  * Pareto curves (**Loss‑Recovered vs. L0**) for all three methods (match compute to Gated‑SAE convention).&#x20;
  * Table: # detected anti‑corr. pairs, # collapsed by SG‑Gated, % width reduction at equal fidelity.

* **Protocol C**:

  * ROC/PR curves and AUC tables by anomaly type.
  * A “latent cards” appendix for top signed‑latents used by the detector.

---

## 8) Why these protocols are convincing

* Protocol A recreates a **canonical geometric phenomenon** (how models arrange correlated and anti‑correlated features in low‑D) and shows our **one‑latent, two‑feature** capability in the clearest possible way, side‑by‑side with baselines.&#x20;
* Protocol B pairs **closed‑form parameter accounting** with **empirical Pareto analyses** under the **Gated‑SAE training recipe** (separate detection/magnitude + frozen‑decoder aux loss + Jump‑ReLU view). This positions our claim within known, peer‑reviewed improvements to SAEs. &#x20;
* Protocol C demonstrates **practical value**—signed latents produce richer statistics (especially **sign asymmetry**) that can flag shifts anomalies that plain ReLU latents miss.

---

### Appendix: Brief literature anchors we referenced

* **Gated SAEs**: separation of detection and magnitude, auxiliary frozen decoder, weight‑tying → Jump‑ReLU; training compute considerations; standard evaluation (Loss‑Recovered vs L0). &#x20;
* **Toy models**: definitions and experimental setup for correlated / anti‑correlated feature sets; 2‑D visualization conventions.&#x20;

---

### Final acceptance gate (across all protocols)

1. **Reproducibility** – all figures regenerate from a single command with fixed seeds; logs include seeds and commit SHA.
2. **Clarity** – each report is self‑contained with method and hyperparameters specified (λ, width, steps, sites).
3. **Evidence** –

   * A: Anti‑correlated pair **collapse** visually and quantitatively for SG‑Gated.
   * B: **Parameter savings** numerically match formulas and **width** required at equal fidelity is **lower** for SG‑Gated when anti‑correlated pairs are prevalent; compute overhead is measured and reported.
   * C: **Detection AUC** meets thresholds and qualitative latent analyses are compelling.

If you follow this plan verbatim, we’ll have: a clean **geometric demo**, a **quantitative capacity audit**, and a **downstream application**—sufficient for a strong experimental section in a top‑tier paper.

