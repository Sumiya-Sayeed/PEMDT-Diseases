# Cirrhosis and the PEMDT Model — What I Tried and What I Found

## The question
I wanted to know whether the PEMDT idea — a probabilistic model that tracks a hidden "health
state" over time from routine labs — is actually useful for cirrhosis, using the MIMIC-IV
hospital dataset. Two angles: can it warn early that a cirrhosis patient is about to get worse
(decompensate), and can it flag cirrhosis coming in people who are still only at risk?

## What I built
Starting from about 8,000 cirrhosis patients (and ~20,000 at-risk patients) in MIMIC-IV, I built
and tested several versions:

- An **early-warning model** for decompensation, using only the first 24 hours of an admission.
- A **latent severity-staging model** that sorts patients into ordered stages from their labs.
- A test of whether **treatments** (lactulose, diuretics, vasopressors) are better thought of as
  *signs* of how sick someone is, or as things that actually *push* the disease forward.
- An **early-detection** track: among at-risk people who aren't cirrhotic yet, can their labs flag
  who will later develop cirrhosis?
- A **progression-to-decompensation** model framed as survival (time-to-event), which turned out
  to be the most important one.
- Two ways of **explaining** the model (an internal likelihood method and SHAP), and a check of
  whether they agree.

## The headline finding
**When I framed the problem as "time to progression to decompensated cirrhosis," the latent
stage clearly and significantly beat the standard lab scores.** This is the result everything
else had been building toward.

Among at-risk patients, the baseline latent stage stratified one-year decompensation risk about
eight-fold — 0.6%, 3.3%, 4.7% across the three stages. On discrimination, the stage-based model
reached an AUC around 0.79 versus 0.68 for MELD-Na and 0.62 for FIB-4, and a **paired test
confirmed the gap is real**: the stage beat MELD-Na by +0.11 (95% CI +0.04 to +0.18) and FIB-4 by
+0.16 (+0.09 to +0.24). A proper survival model agreed — concordance of 0.73 for the stage versus
0.65 for MELD/FIB-4 — and, importantly, in a clean baseline-only model with no look-ahead bias,
the stage stayed an **independent** predictor (hazard ratio ~1.2 per step) even with MELD and
FIB-4 sitting right next to it. So it isn't just repackaging those scores; it carries extra
information. A nice detail: *how long* a patient sits in the advanced stage was the strongest
single signal — something a one-off MELD value can't capture.

## What didn't beat the simple scores (and why that's still useful)
For most of the other framings, a plain validated lab score was statistically as good as the
fancier model — and that's an honest, worth-reporting result:

- **Early warning of decompensation (within already-cirrhotic patients).** The model worked —
  AUC around 0.73, about a two-day median lead time — but MELD-Na matched it in every version.
  Splitting by event type helped (≈0.76 for encephalopathy, ≈0.79 for variceal bleeding), but
  the model still only tied MELD within each subtype.
- **The "treatments as drivers" idea didn't hold up.** It first looked like being on certain
  drugs predicted getting worse, but once I properly adjusted for how sick people already were,
  that signal vanished. It was confounding (the sickest patients get those drugs), not the
  treatment pushing the disease. A clean negative result.
- **Early detection of cirrhosis itself.** Among ~20,000 at-risk patients, about 900 later got a
  cirrhosis diagnosis, and a simple FIB-4 score at their first visit separated them with an AUC
  near 0.70, flagging them roughly two years before the diagnosis was recorded. The honest catch:
  those patients already had fairly advanced fibrosis at baseline — so this is catching cirrhosis
  that's *already there but not yet diagnosed*, not catching it early and silent. Useful for
  case-finding, but not screening.

## On explaining the model
I compared the model's built-in explanation against SHAP. For the decompensation models they
agreed well (rank agreement ~0.85–0.93) once I corrected a scaling quirk in the built-in method,
and both pointed to the expected drivers — bilirubin, INR, creatinine (the MELD ingredients).
For the early fibrosis model they disagreed, because platelets behave oddly across the stages
(useful for prediction but not cleanly separated between stages). Practical takeaway: SHAP is the
safer default, with the corrected internal method as a per-patient companion.

## The big picture
The honest story has two halves. For *snapshot* tasks — is this patient sick right now, will
they decompensate in the next day or two — simple validated scores like MELD-Na and FIB-4 are
genuinely hard to beat, and a lot of the fancy machinery just re-encodes them. But when I framed
the question as **progression over time to decompensation**, and gave the latent stage a richer
set of labs plus its own trajectory (how long it spends in the advanced state), the PEMDT model
added **real, statistically significant prognostic value beyond MELD and FIB-4**. That's the
clearest win, and it fits the model's strength: it's built to track a state over time, not to
beat a static score at a single moment.

The main limitation throughout isn't the model, it's the data. MIMIC is a hospital/ICU snapshot
with short, sparse follow-up, so survival estimates are noisier and "early detection" is really
"earlier than the diagnosis code." If I take this further, the progression-to-decompensation and
fibrosis-progression angle is the one with both the strongest result and the clearest clinical
story — ideally validated on a dataset with proper long-term follow-up.
