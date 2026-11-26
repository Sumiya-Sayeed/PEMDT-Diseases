"""Shared evaluation utilities for the IBMED-D-26-00792 revision analyses.

Used by:
  PEMDT-AKI/AKI_revision_IBMED.ipynb
  PEMDT - Respiratory/RF_revision_IBMED.ipynb
  PEMDT-cirrhosis/PEMDT_cirrhosis_progression_hmm.ipynb (REVISION cells)

Design notes
  * Bootstrap CIs resample PATIENTS (subjects), never rows, so that the many
    correlated hours or windows of one patient move together.
  * The bootstrap is implemented with per-row multiplicity weights on a single
    presorted array, which gives the same estimator as rebuilding the resampled
    data frame but avoids re-sorting millions of rows in every replicate.
  * Calibration intercept (calibration-in-the-large, slope fixed at 1) and
    calibration slope follow the logistic recalibration framework
    (Cox 1958; Van Calster et al. 2019). Both are fitted by IRLS here so that
    no statsmodels dependency is needed.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_trapz = getattr(np, "trapezoid", None) or np.trapz

# --------------------------------------------------------------------------
# Loading definitions from the original notebooks (no analyses are re-run)
# --------------------------------------------------------------------------

def cell_source(nb_path: Path | str, marker: str) -> str:
    """Source of the single code cell in `nb_path` that contains `marker`."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    hits = [c for c in nb["cells"] if c["cell_type"] == "code"
            and marker in "".join(c["source"])]
    if len(hits) != 1:
        raise RuntimeError(f"marker {marker!r} matched {len(hits)} code cells in {nb_path}")
    return "".join(hits[0]["source"])


def defs_only(src: str) -> str:
    """Keep only imports, function and class definitions of a cell (drop top-level runs)."""
    lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!"))]
    tree = ast.parse("\n".join(lines))
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def strip_main_guard(src: str) -> str:
    """Remove an `if __name__ == "__main__":` block (and everything after it)."""
    key = 'if __name__ == "__main__":'
    return src.split(key)[0] if key in src else src


# --------------------------------------------------------------------------
# Weighted metrics on presorted scores (exact sklearn-equivalent definitions)
# --------------------------------------------------------------------------

class SortedScores:
    """Scores sorted once (descending) with tie blocks, for fast weighted AUROC/AP."""

    def __init__(self, p: np.ndarray):
        p = np.asarray(p, float)
        self.order = np.argsort(-p, kind="mergesort")
        ps = p[self.order]
        self.last = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]


def w_auc_ap(y: np.ndarray, w: np.ndarray, ss: SortedScores) -> Tuple[float, float]:
    """Weighted AUROC (ties count 1/2) and average precision (sklearn definition)."""
    yo = y[ss.order].astype(float)
    wo = w[ss.order]
    tp = np.cumsum(wo * yo)[ss.last]
    fp = np.cumsum(wo * (1.0 - yo))[ss.last]
    P, N = tp[-1], fp[-1]
    if P <= 0 or N <= 0:
        return np.nan, np.nan
    auc = float(_trapz(np.r_[0.0, tp / P], np.r_[0.0, fp / N]))
    prec = tp / np.maximum(tp + fp, 1e-300)
    rec = tp / P
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return auc, ap


def _irls(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None,
          offset: Optional[np.ndarray] = None, beta0: Optional[np.ndarray] = None,
          iters: int = 50) -> np.ndarray:
    """Weighted logistic regression by IRLS (unpenalized)."""
    n, k = X.shape
    w = np.ones(n) if w is None else w
    off = 0.0 if offset is None else offset
    beta = np.zeros(k) if beta0 is None else np.array(beta0, float)

    def nll(b):
        eta = X @ b + off
        return float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)))

    cur = nll(beta)
    for _ in range(iters):
        eta = np.clip(X @ beta + off, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = w * mu * (1.0 - mu) + 1e-12
        g = X.T @ (w * (y - mu))
        H = (X * W[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(k), g)
        t = 1.0                                   # step-halving keeps IRLS monotone
        while t > 1e-6:
            new = nll(beta + t * step)
            if new <= cur + 1e-10:
                break
            t *= 0.5
        beta = beta + t * step
        done = abs(cur - new) < 1e-9 * (abs(cur) + 1.0) or np.max(np.abs(t * step)) < 1e-8
        cur = new
        if done:
            break
    return beta


def calibration_stats(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Brier, calibration intercept (CITL, slope fixed at 1), calibration slope, E/O."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    w = np.ones_like(y) if w is None else w
    lp = np.log(p / (1 - p))
    sw = w.sum()
    brier = float(np.sum(w * (p - y) ** 2) / sw)
    eo = float(np.sum(w * p) / max(np.sum(w * y), 1e-12))
    slope = float(_irls(np.c_[np.ones_like(lp), lp], y, w)[1])
    citl = float(_irls(np.ones((len(y), 1)), y, w, offset=lp)[0])
    return dict(brier=brier, citl=citl, slope=slope, eo=eo)


METRICS = ["AUROC", "AUPRC", "Brier", "CITL", "Slope", "E/O"]


def metric_vector(y, p, w, ss) -> np.ndarray:
    auc, ap = w_auc_ap(y, w, ss)
    c = calibration_stats(y, p, w)
    return np.array([auc, ap, c["brier"], c["citl"], c["slope"], c["eo"]])


class PatientBootstrap:
    """Patient-level bootstrap through per-row multiplicity weights."""

    def __init__(self, subject: np.ndarray, n_boot: int = 200, seed: int = 7):
        _, self.sidx = np.unique(np.asarray(subject), return_inverse=True)
        self.ns = int(self.sidx.max()) + 1
        self.n_boot, self.seed = n_boot, seed

    def weights(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_boot):
            c = np.bincount(rng.integers(0, self.ns, self.ns), minlength=self.ns)
            yield c[self.sidx].astype(float)


def evaluate(y, p, subject, n_boot=200, seed=7, alpha=0.05, label="",
             rank_score=None) -> pd.DataFrame:
    """Point estimate + patient-bootstrap CI for AUROC, AUPRC, Brier, CITL, slope, E/O.

    `p` is the probability used for calibration metrics. `rank_score` (optional) is the
    score used for AUROC/AUPRC; pass the raw belief score when `p` is a recalibrated
    version of it, so that floating-point ties created by the recalibration map do not
    alter discrimination (a monotone map cannot change AUROC in exact arithmetic)."""
    y = np.asarray(y, int); p = np.asarray(p, float)
    r = p if rank_score is None else np.asarray(rank_score, float)
    ok = np.isfinite(p) & np.isfinite(r)
    y, p, r, subject = y[ok], p[ok], r[ok], np.asarray(subject)[ok]
    ss = SortedScores(r)
    point = metric_vector(y, p, np.ones(len(y)), ss)
    boots = []
    for w in PatientBootstrap(subject, n_boot, seed).weights():
        if np.sum(w * y) == 0 or np.sum(w * (1 - y)) == 0:
            continue
        boots.append(metric_vector(y, p, w, ss))
    B = np.array(boots)
    lo, hi = np.nanpercentile(B, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    out = pd.DataFrame({"model": label, "metric": METRICS, "estimate": point, "lo": lo, "hi": hi,
                        "n": len(y), "events": int(y.sum()), "patients": len(np.unique(subject))})
    return out


def paired_delta_auc(y, p_a, p_b, subject, n_boot=200, seed=7, alpha=0.05) -> Tuple[float, float, float]:
    """AUROC(a) - AUROC(b) on identical rows, patient-bootstrap CI."""
    y = np.asarray(y, int)
    ok = np.isfinite(p_a) & np.isfinite(p_b)
    y, p_a, p_b, subject = y[ok], np.asarray(p_a)[ok], np.asarray(p_b)[ok], np.asarray(subject)[ok]
    sa, sb = SortedScores(p_a), SortedScores(p_b)
    one = np.ones(len(y))
    point = w_auc_ap(y, one, sa)[0] - w_auc_ap(y, one, sb)[0]
    d = [w_auc_ap(y, w, sa)[0] - w_auc_ap(y, w, sb)[0]
         for w in PatientBootstrap(subject, n_boot, seed).weights()]
    lo, hi = np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def wide(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """estimate (lo to hi) strings, one row per model."""
    # Plain loop instead of DataFrame.pivot: pivot/unstack on string columns fails in some
    # pandas versions ("'tuple' object has no attribute 'T'"). Keeps the input model order.
    table: Dict[str, Dict[str, str]] = {}
    for model, metric, est, lo, hi in zip(df["model"], df["metric"], df["estimate"], df["lo"], df["hi"]):
        table.setdefault(str(model), {})[str(metric)] = (
            f"{float(est):.{digits}f} ({float(lo):.{digits}f} to {float(hi):.{digits}f})")
    t = pd.DataFrame.from_dict(table, orient="index")
    t.index.name = "model"
    return t[[m for m in METRICS if m in t.columns]]


# --------------------------------------------------------------------------
# Recalibration
# --------------------------------------------------------------------------

def platt_fit(score_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    s = np.asarray(score_train, float)
    return _irls(np.c_[np.ones_like(s), s], np.asarray(y_train, float))


def platt_apply(beta: np.ndarray, score: np.ndarray) -> np.ndarray:
    s = np.asarray(score, float)
    return 1.0 / (1.0 + np.exp(-np.clip(beta[0] + beta[1] * s, -30, 30)))


# --------------------------------------------------------------------------
# Operating characteristics and decision curves
# --------------------------------------------------------------------------

def confusion_at(y, p, thr) -> Dict[str, float]:
    y = np.asarray(y, int); a = np.asarray(p) >= thr
    tp = int((a & (y == 1)).sum()); fp = int((a & (y == 0)).sum())
    fn = int((~a & (y == 1)).sum()); tn = int((~a & (y == 0)).sum())
    return dict(threshold=float(thr), TP=tp, FP=fp, FN=fn, TN=tn,
                sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                PPV=tp / max(tp + fp, 1), NPV=tn / max(tn + fn, 1),
                flagged_frac=float(a.mean()))


def threshold_for_specificity(y, p, spec=0.90) -> float:
    """Smallest threshold whose specificity on (y, p) is >= spec (choose on TRAIN data)."""
    neg = np.sort(np.asarray(p)[np.asarray(y) == 0])
    return float(neg[int(np.ceil(spec * len(neg))) - 1]) + 1e-12


def threshold_for_sensitivity(y, p, sens=0.80) -> float:
    pos = np.sort(np.asarray(p)[np.asarray(y) == 1])[::-1]
    return float(pos[int(np.ceil(sens * len(pos))) - 1])


def net_benefit(y, p, thresholds: Iterable[float]) -> pd.DataFrame:
    """Decision-curve net benefit (Vickers and Elkin, Med Decis Making 2006)."""
    y = np.asarray(y, int); n = len(y); prev = y.mean()
    rows = []
    for pt in thresholds:
        a = np.asarray(p) >= pt
        tp = (a & (y == 1)).sum(); fp = (a & (y == 0)).sum()
        rows.append(dict(threshold=pt, net_benefit=tp / n - fp / n * pt / (1 - pt),
                         treat_all=prev - (1 - prev) * pt / (1 - pt)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Survival helpers (cirrhosis)
# --------------------------------------------------------------------------

def km_survival(time: np.ndarray, event: np.ndarray):
    """Kaplan-Meier S(t) as a right-continuous step function evaluator."""
    t = np.asarray(time, float); e = np.asarray(event, int)
    ut = np.unique(t[e == 1])
    s, surv = 1.0, []
    for u in ut:
        d = ((t == u) & (e == 1)).sum(); r = (t >= u).sum()
        s *= (1 - d / r) if r > 0 else 1.0
        surv.append(s)
    ut, surv = np.asarray(ut), np.asarray(surv)

    def S(x):
        x = np.atleast_1d(np.asarray(x, float))
        i = np.searchsorted(ut, x, side="right") - 1
        return np.where(i >= 0, surv[np.clip(i, 0, None)] if len(surv) else 1.0, 1.0)
    return S


def ipcw_auc(time, event, score, horizon: float) -> float:
    """Cumulative/dynamic AUC(t) with inverse-probability-of-censoring weights.

    Cases: event at or before `horizon` (weight 1/G(T_i-)); controls: still event-free
    after `horizon` (weight 1/G(horizon)). G is the Kaplan-Meier censoring survival.
    Patients censored before `horizon` contribute through the weights only
    (Uno et al., JASA 2007; Blanche et al., Stat Med 2013).
    """
    t = np.asarray(time, float); e = np.asarray(event, int); s = np.asarray(score, float)
    ok = np.isfinite(s); t, e, s = t[ok], e[ok], s[ok]
    G = km_survival(t, 1 - e)
    case = (t <= horizon) & (e == 1)
    ctrl = t > horizon
    if case.sum() == 0 or ctrl.sum() == 0:
        return np.nan
    wc = 1.0 / np.maximum(G(t[case] - 1e-9), 1e-12)
    wk = 1.0 / max(float(G(horizon)[0]), 1e-12)
    sc, sk = s[case], np.sort(s[ctrl])
    below = np.searchsorted(sk, sc, side="left")
    ties = np.searchsorted(sk, sc, side="right") - below
    conc = (below + 0.5 * ties) * wk
    return float(np.sum(wc * conc) / (wc.sum() * wk * len(sk)))


def ipcw_auc_boot(time, event, score, horizon, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    t = np.asarray(time, float); e = np.asarray(event, int); s = np.asarray(score, float)
    point = ipcw_auc(t, e, s, horizon)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(t), len(t))
        v = ipcw_auc(t[i], e[i], s[i], horizon)
        if np.isfinite(v):
            b.append(v)
    lo, hi = np.percentile(b, [2.5, 97.5]) if b else (np.nan, np.nan)
    return point, float(lo), float(hi)


def harrell_c(time, event, score) -> float:
    """Harrell's C (higher score = higher risk = shorter time)."""
    t = np.asarray(time, float); e = np.asarray(event, int); s = np.asarray(score, float)
    ok = np.isfinite(s); t, e, s = t[ok], e[ok], s[ok]
    num = den = 0.0
    order = np.argsort(t)
    t, e, s = t[order], e[order], s[order]
    for i in np.nonzero(e == 1)[0]:
        later = t > t[i]
        if not later.any():
            continue
        den += later.sum()
        num += (s[i] > s[later]).sum() + 0.5 * (s[i] == s[later]).sum()
    return float(num / den) if den > 0 else np.nan


def harrell_c_boot(time, event, score, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    t = np.asarray(time, float); e = np.asarray(event, int); s = np.asarray(score, float)
    point = harrell_c(t, e, s)
    b = [harrell_c(t[i], e[i], s[i]) for i in (rng.integers(0, len(t), len(t)) for _ in range(n_boot))]
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

def env_versions() -> Dict[str, str]:
    import platform, sys
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ["numpy", "pandas", "scipy", "sklearn", "jax", "xgboost", "torch", "lifelines"]:
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:
            out[m] = "not installed"
    return out


def anchor_year_groups(patients_csv: Path | str) -> pd.Series:
    """subject_id -> anchor_year_group (MIMIC-IV hosp/patients)."""
    p = pd.read_csv(patients_csv, usecols=["subject_id", "anchor_year_group"])
    return p.set_index("subject_id")["anchor_year_group"]


def temporal_split(groups: pd.Series, min_test_frac: float = 0.10) -> Tuple[List[str], List[str]]:
    """Most recent anchor_year_group(s) form the temporal test set (>= min_test_frac of subjects)."""
    order = sorted(groups.dropna().unique(), key=lambda g: int(str(g)[:4]))
    counts = groups.value_counts()
    test: List[str] = []
    for g in reversed(order):
        test.append(g)
        if counts[test].sum() / counts.sum() >= min_test_frac:
            break
    dev = [g for g in order if g not in test]
    return dev, sorted(test)
