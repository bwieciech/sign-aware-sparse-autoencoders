# Sign-Aware Gated Sparse Autoencoders (SA-GSAE)

We formalize a **sign-aware** extension of Gated SAEs that enables a *single latent* to carry **two-sided evidence** (positive or negative) along one learned axis while preserving the core gated-training advantages (no shrinkage, better L0–fidelity Pareto).

## 1. Setting and baseline

Let $x\in\mathbb{R}^n$ be an activation vector from a fixed site in a pretrained network. A dictionary $D=[d_1,\dots,d_M]\in\mathbb{R}^{n\times M}$ and sparse encoder $f:\mathbb{R}^n\to\mathbb{R}^M$ reconstruct $x\approx Df(x)+b_0$. In a baseline SAE,

$$
f(x)=\mathrm{ReLU}\!\big(W_{\text{enc}}(x-b_0)+b_{\text{enc}}\big),\quad 
\hat x=D\,f(x)+b_0,
$$

optimized via $\|x-\hat x\|_2^2+\lambda\|f(x)\|_1$ with unit-norm decoder columns $\|D_{\cdot i}\|_2=1$. The $\ell_1$ penalty on *magnitudes* induces **shrinkage**.

Gated SAEs decouple *detection* from *magnitude*: a binary gate chooses which atoms fire, while magnitudes are regressed without $\ell_1$. With weight-sharing,

$$
\pi_{\text{gate}}(x)=W_{\text{gate}}(x-b_0)+b_{\text{gate}},\qquad
W_{\text{mag}}=\mathrm{diag}\!\big(e^{r_{\text{mag}}}\big)\,W_{\text{gate}},
$$

the encoder collapses to a single linear map followed by a **Jump-ReLU**: one thresholded tail with learned slope and a true “off” region. The loss keeps $\ell_1$ *only on gate preactivations* and adds a **frozen-decoder via-gate** auxiliary reconstruction to train the gate without biasing $D$.

## 2. Our extension: signed latents with a two-sided Jump-ReLU

### 2.1 Parameterization

For unit $i$, let $u_i(x)=w_i^\top(x-b_0)$ and $\pi_i(x)=u_i(x)+b_{{\rm enc},i}$. Introduce **asymmetric dead-zone thresholds** $\delta_i^+>0,\delta_i^->0$, **two slopes** $e^{r_i^+},e^{r_i^-}$, and a shared magnitude bias $b_{{\rm mag},i}$. Define

$$
g_i^+(x)=\mathbf{1}\!\{\pi_i(x)>\delta_i^+\},\quad
g_i^-(x)=\mathbf{1}\!\{\pi_i(x)<-\delta_i^-\},
$$

$$
z_i^+(x)=e^{r_i^+}u_i(x)+b_{{\rm mag},i},\quad
z_i^-(x)=e^{r_i^-}u_i(x)+b_{{\rm mag},i}.
$$

The **signed latent** is

$$
\boxed{a_i(x)=g_i^+(x)\,\mathrm{ReLU}\!\big(z_i^+(x)\big)\;-\;g_i^-(x)\,\mathrm{ReLU}\!\big(-z_i^-(x)\big)}\tag{S1}
$$

and the reconstruction is $\hat x=D\,a(x)+b_0$ with $\|D_{\cdot i}\|_2=1$.

**Interpretation.** (S1) is a *two-sided* Jump-ReLU on *one axis* $u_i$: a hard dead-zone $[-\delta_i^-,\delta_i^+]$ and two linear tails of opposite sign with independent slopes. Because the negative and positive gates are mutually exclusive ($\pi>\delta^+$ and $\pi<-\delta^-$ cannot co-occur), spurious cancellations are structurally suppressed.

### 2.2 Training objective

We preserve gated training: $\ell_1$ on the gate only; the decoder is trained from the main reconstruction and *not* touched by the auxiliary via-gate term.

$$
\boxed{
\mathcal{L}(x)
= \underbrace{\|x-D\,a(x)-b_0\|_2^2}_{\text{reconstruction}}
\;+\;\lambda\,\underbrace{\sum_i\Big[\mathrm{ReLU}\big(\pi_i(x)-\delta_i^+\big)+\mathrm{ReLU}\big(-\pi_i(x)-\delta_i^-\big)\Big]}_{\text{sparsity on signed gate}}
\;+\;\underbrace{\big\|x-D^{\rm frozen}v(x)-b_0^{\rm frozen}\big\|_2^2}_{\text{via-gate}}
}\tag{S2}
$$

with continuous **via-gate** proxy

$$
v_i(x)=\mathrm{ReLU}\big(\pi_i(x)-\delta_i^+\big)-\mathrm{ReLU}\big(-\pi_i(x)-\delta_i^-\big).
$$

**Gradient flow.** The $\ell_1$ term pushes $\pi$ *into* the dead-zone; the main MSE trains $D$ and the magnitude path; the via-gate term supplies stable gradients to gate parameters while the decoder is stop-grad.

### 2.3 Why this helps (and when)

* **Pair-collapse of anticorrelated features.** If two non-co-occurring features lie on $\pm d$, nonnegative SAEs need two atoms $\{d,-d\}$. SA-GSAE uses **one** axis with opposite signs, halving the *width* required for those pairs and reducing L0 at matched fidelity when such pairs are common.
* **No shrinkage preserved.** Magnitudes are not $\ell_1$-penalized; the signed variant inherits the gated SAE’s “no-shrinkage” behavior for active coefficients.
* **Better inductive bias for directional evidence.** Many phenomena are naturally two-sided (positive/negative sentiment; up/down deviations; A vs B). A dead-zone with two tails matches this structure.

## 3. Piecewise-linear view

As a function of $\pi_i$ and $u_i$,

$$
a_i(x)=
\begin{cases}
0,& |\pi_i(x)|\le \delta_i^{\pm},\\
\mathrm{ReLU}(e^{r_i^+}u_i+b_{{\rm mag},i}),& \pi_i(x)>\delta_i^+,\\
-\mathrm{ReLU}(-e^{r_i^-}u_i-b_{{\rm mag},i}),& \pi_i(x)<-\delta_i^-.
\end{cases}
$$

With the shared-direction tie $W^{\pm}_{\mathrm{mag}}=\mathrm{diag}(e^{r^\pm})W_{\mathrm{gate}}$, each latent remains a single projected **bi-Jump-ReLU**, a direct signed generalization of the one-sided Jump-ReLU view.

## 4. Capacity & compute accounting (sanity-checked)

Let $d_{\text{in}}=n$ and suppose a fraction $p$ of baseline atoms belong to $\pm d$ pairs.

**Width.** Signed model needs

$$
H_{\pm}=H\,(1-\tfrac{p}{2}).
$$

**Parameters.** Baseline $P_{\text{base}}=2Hn+H+n$. Our model adds $+6H_{\pm}$ scalars (two offsets $\delta^\pm$, two slopes $r^\pm$, shared $b_{\rm mag}$, and the unsigned $r_{\rm mag}$). Thus

$$
P_{\pm}=2H_{\pm}n+H_{\pm}+n+6H_{\pm},
$$

so

$$
P_{\pm}-P_{\text{base}}=H\big(6-p(n+3.5)\big).
$$

**Net saving whenever** $p>\frac{6}{n+3.5}$ (e.g., threshold $\approx0.6\%$ at $n=1024$; $\approx0.15\%$ at $n=4096$). ✔️

**FLOPs (inference).** Compute $t=W_{\text{enc}}(x-b_0)$ *once*, then form $z^\pm$ by elementwise ops. Encoder+decoder matmuls scale with width:

$$
\text{ratio}_{\text{infer}}=1-\tfrac{p}{2}.
$$

**FLOPs (training).** The via-gate adds one extra decoder pass; empirically, overall loss compute is $\approx 1.5\times$ a matched-width baseline. Width reduction gives $\text{ratio}_{\text{train}}\approx 1.5(1-\tfrac{p}{2})$, which dips **below baseline** when $p>2/3$. In practice, activation I/O often dominates, so wall-clock slow-down is smaller.

## 5. Experimental protocols (paper-ready, model-agnostic)

We propose three complementary experiments designed to isolate SA-GSAE’s **unique** benefits. Each uses the same training recipe: unit-norm decoder columns, $\ell_1$ on gate *only*, frozen-decoder via-gate, and dead-latent resampling.

### Protocol A — *Polarity dial* (asymmetry along one axis)

**Data.** 2D synthetic with a single direction $d$. Draw positive/negative sides with **different** magnitude laws (e.g., LogNormal vs Exponential) and additive isotropic noise.

**Models.** Baseline SAE, unsigned Gated SAE, **SA-GSAE**.

**Metrics.** Global MSE; per-side errors $\text{MSE}_{+},\text{MSE}_{-}$; per-side shrinkage factors (best-fit scale $\gamma_\pm$); L0 at matched fidelity.

**Expectation.** SA-GSAE reduces $\max(\text{MSE}_+,\text{MSE}_-)$ by 10–30% vs unsigned gated when side statistics differ, keeping $\gamma_+\approx\gamma_-\approx1$.

### Protocol B — *Two-sided anomaly detector* (one latent, two tails)

**Data.** Time-series/tabular with symmetric deviations (too-high / too-low). Train on *normal* only.

**Detector.** Threshold gate preactivations separately for $\pm$ tails; aggregate across latents.

**Metrics.** AUROC/AUPRC per-tail & overall; balanced TPR at fixed FPR; **latents required** to reach a target AUROC.

**Expectation.** At equal AUROC, SA-GSAE needs ≈½ as many latents for one-axis two-tail anomalies; at equal width, improves balanced TPR by 5–15%.

### Protocol C — *Toy anticorrelation geometry* (one latent for a pair)

**Data.** Synthetic “Toy Models of Superposition”-style with (i) **anticorrelated** pairs (non-co-occurring on $\pm d$), optionally (ii) a correlated pair. Constrain the SAE to **2 hidden dims** and plot decoder columns on the unit circle.

**Measurements.** Geometry (points on circle): baseline/unsigned gated show **two antipodal** atoms per anticorrelated pair; SA-GSAE shows **one** atom (two signs). One-latent clustering score (NMI/V-measure) for the pair; width required vs MSE.

**Expectation.** Clear visual collapse (2→1 atom per pair) and ≈2× width reduction in anticorrelation-dominated regimes.

## 6. Practical notes

* **Avoid duplicated matmuls.** Cache $t=W_{\text{enc}}(x-b_0)$; form $z^\pm$ via elementwise ops.
* **Asymmetry priors.** Initialize $r^+=r^-$, $\delta^+=\delta^-$; relax only if data suggest.
* **Interpretability.** Report per-latent *two-sided* examples (top-$k$ positive and negative) and activation histograms showing the dead-zone.

## 7. Limitations & validity checks

* **Cancellation.** While mutual exclusivity prevents same-latent co-firing, cross-latent cancellations remain possible—as in all linear decoders. Monitoring near-zero $|a_i|$ with large opposing contributions is recommended.
* **Sign/atom identifiability.** As usual, $D_{\cdot i}\to -D_{\cdot i}$ and $a_i\to -a_i$ leave reconstructions unchanged; we rely on sign-split reporting for interpretability.
* **Compute claim scope.** The $1.5\times$ training-compute factor reflects the added decoder pass under typical activation-I/O bottlenecks; exact ratios depend on implementation details.
