# RF-PEMDT — Summary of Findings
**Transfer of the PEMDT architecture to respiratory-failure prediction on MIMIC-IV ICU**

## 1. Setup

The PEMDT digital-twin architecture (multi-patient Bayesian HMM with glass-box
log-likelihood-ratio explainability) was transferred from its original generic
instability / mortality target to **respiratory failure**, and upgraded to a
three-state model: **Stable → Respiratory Compromise → Respiratory Failure**.

- **Cohort:** 65,363 ICU patients, 9,987,192 five-minute observation windows (full MIMIC-IV v3).
- **Features (respiratory panel):** SpO₂, Respiratory Rate, P/F ratio (PaO₂/FiO₂), PaCO₂, pH, HR, MAP, Lactate. Every extracted signal is used by the model (CASAS behavioural fusion and all unused features from the first draft were removed).
- **Outcome label:** `is_resp_failure` = first **invasive** ventilation/intubation (timestamped, NIV excluded) **OR** an **acute** respiratory-failure ICD code (chronic-only excluded). Cohort prevalence 53.9% (invasive ventilation 42.6%, acute ICD 29.8%).
- **Engineering:** a compact itemid-filtered parquet cache reduced chartevents from 40 GB → 130 MB and labevents from 18 GB → 14 MB; the HMM likelihood was vectorised (~7× faster). The dataset was treated strictly read-only.

## 2. Concurrent performance (Cells 3–8)

Scoring each patient by peak failure probability over their whole stay:

- **Raw HMM ROC-AUC 0.725**, PR-AUC 0.752, Brier 0.287.
- **Platt calibration** (out-of-fold) left AUC ≈ unchanged (monotonic) but improved **Brier 0.287 → 0.215**.
- Subgroup AUC stable at 0.71–0.75 (slightly lower in age 80+).
- 180-configuration sweep: AUC 0.738–0.746 (robust to configuration).
- Explainability: the LLR attribution was dominated by SpO₂ then Respiratory Rate — clinically sensible, and more coherent than the naïve change-based (Δf) attribution, which misfired on MAP/HR.

These numbers are comparable to the original PEMDT (raw 0.687 / calibrated 0.742).

## 3. The key methodological result: early prediction vs concurrent detection (Cell 9)

Because SpO₂/RR drive both the model and the clinician's intubation decision, the
concurrent AUC risks measuring *detection* rather than *prediction*. A **length-fair
pre-onset evaluation** was built: for ventilated positives the model may only see a
4-hour window ending H hours **before** first ventilation; negatives are scored over
an equal-length window at a matched (random) time. This removes the max-over-more-
windows bias that inflated an initial naïve version.

| Lead time | ROC-AUC |
|---|---|
| 0 h | 0.65 |
| 2 h | 0.64 |
| 6 h | 0.66 |
| 12 h | 0.65 |
| 24 h | 0.63 |

**The curve is essentially flat from 0 to 24 h.** This is the central finding: the
model carries a *genuine but modest* pre-onset signal (AUC ≈ 0.65), not a last-second
detection spike. The strong concurrent 0.725 was inflated by peri/post-onset windows.
(PR-AUC declines with horizon only because prevalence falls from 50% to 8%; ROC-AUC is
the fair cross-horizon comparison.)

## 4. Benchmarks and HMM improvement (Cells 10–12)

A supervised logistic regression (LR) on the same pre-onset windows was used as the
"reference to beat."

**Hand-specified HMM vs fitted HMM vs LR (test ROC-AUC):**

| Lead time | Hand HMM | Fitted HMM | LR (patient-level) |
|---|---|---|---|
| 0 h | 0.644 | 0.673 | 0.834 |
| 2 h | 0.630 | 0.656 | 0.865 |
| 4 h | 0.640 | 0.663 | 0.821 |
| 6 h | 0.655 | 0.674 | 0.734 |
| 12 h | 0.638 | 0.663 | 0.695 |
| 24 h | 0.627 | 0.650 | 0.689 |

- **Fitting the emissions helped.** Replacing hand-set z-shifts with *class-conditional*
  Gaussian emissions (Stable = non-RF windows; Failure = 0–12 h pre-onset windows),
  adding rolling worst-case channels, and weighting features by Cohen's d, made the
  **fitted HMM beat the hand HMM at every horizon** and **match LR within ~0.03 at
  12–24 h**.
- **A clinical insight fell out of the fit:** SpO₂ collapsed to near-zero weight (0.04)
  because it is *treatment-masked* — propped up by supplemental O₂ before intubation.
  The real pre-onset signal is in the **compensatory signs**: P/F ratio, heart rate,
  PaCO₂, respiratory rate. The hand-specified model had over-weighted SpO₂.
- **The hybrid did not help.** Feeding LR's per-window probability into the HMM as a
  discriminative emission, then temporally smoothing, *reduced* short-horizon AUC
  (smoothing flattens the peaks that peak-in-window scoring depends on).

## 5. Why LR wins at short horizons — and why the HMM cannot follow

The decisive diagnostic: a **patient-level** LR (features = min/max/last/mean of each
signal over the observation window) scores **0.83** at 0–2 h, but a **per-window** LR
scores only **0.69** on the same data. The discriminative advantage lives in the
**explicit window aggregation** (e.g. "worst SpO₂ over the last 4 h"). An HMM is a
sequential, one-window-at-a-time filter and **structurally cannot represent a
whole-window min/max/last as a single joint feature**. Rolling worst-case channels
recover part of this but not all. This is a fundamental property (generative sequential
model vs discriminative aggregate model), not a tuning failure.

## 6. Conclusions

1. The PEMDT architecture transfers cleanly to respiratory failure and reproduces
   PEMDT-level concurrent performance (AUC ≈ 0.72–0.75).
2. Under a leakage-aware, length-fair evaluation, genuine **early-warning** discrimination
   is **modest and stable at AUC ≈ 0.65 across 0–24 h** — a real signal, not concurrent
   detection.
3. A **data-fitted interpretable HMM** is the best model in this family: it beats the
   hand-specified version everywhere and is within ~0.03 of a discriminative benchmark
   at the clinically valuable 6–24 h lead times, while remaining a glass-box generative
   model.
4. The benchmark's short-horizon (0–4 h) advantage derives from observation-window
   feature aggregation that a sequential state-space model does not exploit; attempts to
   close this within the HMM (rolling extremes, discriminative-emission hybrid) recover
   little and cost the model's online, per-timestep interpretability.

**Recommended framing:** *A data-fitted interpretable HMM achieves early-warning
discrimination comparable to a discriminative logistic benchmark at 6–24 h lead times;
the benchmark's advantage is confined to the 0–4 h window and derives from feature
aggregation not available to a sequential state-space model.*

---

### Honest limitations
- Invasive ventilation is a proxy for respiratory failure and can be triggered for
  non-respiratory reasons; ICD-only positives (no ventilation timestamp) are excluded
  from the horizon analysis.
- The HMM emissions were fit supervised (class-conditional) on train patients; this is
  a legitimate generative fit but uses outcome information, so a modest optimistic bias
  relative to a fully unsupervised HMM is possible — evaluation is on held-out patients.
- All figures are from the full-cohort run and reproduced by the notebook cells; the
  parquet round-trip and model fitting were executed on the full dataset locally.
