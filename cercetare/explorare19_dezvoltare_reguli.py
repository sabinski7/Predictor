"""Explorare 19: variante ale strategiei „numerele celor 10 reguli cu cele mai scurte pauze”.
Verificare pe jurnalul tău (propunerile reale) și pe istoricul lung (propunerile notebook-ului reconstituite,
31.000+ ture, împărțit în bucăți de 500 de ture ca să semene cu jurnalul: statisticile regulilor pornesc de la zero
la fiecare bucată, iar strategia joacă de la a 50-a tură a bucății).
Măsura care contează: eficiența = nimeriri / (numere jucate × 5%). 1,00 = exact hazardul.
Rulare din rădăcina repo-ului: python cercetare/explorare19_dezvoltare_reguli.py"""
import sys, json, contextlib, io
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
from scipy.stats import norm
sys.path.insert(0, "automat")
import pauze

C = pauze.C                       # (2868, 9) coeficienți; regula = bază*20 + c
NB = len(C); NR = NB * 20
START, W = 50, 100


def strategies(P, y, block=None):
    """Întoarce dict nume -> listă (set numere jucate, rezultat) pentru fiecare tură jucată."""
    out = {}
    def add(name, nums, res):
        out.setdefault(name, []).append((nums, res))
    n = len(y)
    t0 = 0
    while t0 < n:
        t1 = n if block is None else min(n, t0 + block)
        run = np.zeros(NR, np.int32); mx = np.zeros(NR, np.int32); hits = np.zeros(NR, np.int32)
        recent = np.zeros(NR, np.int32); hist = []
        for t in range(t0, t1):
            base = P[t] @ C.T                              # (2868,)
            pred = (base[:, None] + np.arange(20) - 1) % 20 + 1   # (2868, 20) numărul prezis de fiecare regulă
            pred = pred.ravel()
            if t - t0 >= START:
                eff = np.maximum(mx, run)
                key = eff.astype(np.int64) * 100000 - hits
                def topk(k, key=key):
                    idx = np.argpartition(key, k)[:k]
                    return idx[np.argsort(key[idx], kind="stable")]
                t10, t20 = topk(10), topk(20)
                v10 = pred[t10]
                add("A. top 10 pauze scurte (actual)", set(v10), y[t])
                add("B. top 5 pauze scurte", set(pred[t10[:5]]), y[t])
                add("C. top 20 pauze scurte", set(pred[t20]), y[t])
                vals, cnt = np.unique(v10, return_counts=True)
                if (cnt >= 2).any():
                    add("D. doar numerele cu 2+ voturi din top 10", set(vals[cnt >= 2]), y[t])
                add("E. top 10 după cele mai multe nimeriri", set(pred[np.argpartition(-hits, 10)[:10]]), y[t])
                add(f"F. top 10 „fierbinți” (nimeriri în ultimele {W} ture)", set(pred[np.argpartition(-recent, 10)[:10]]), y[t])
                due = t20[np.argsort(-(run[t20] / np.maximum(eff[t20], 1)), kind="stable")[:10]]
                add("G. din top 20, cele 10 cele mai „restante”", set(pred[due]), y[t])
                fresh = t20[np.argsort(run[t20], kind="stable")[:10]]
                add("H. din top 20, cele 10 care au nimerit cel mai recent", set(pred[fresh]), y[t])
            # actualizare cu rezultatul turei t
            h = np.arange(NB) * 20 + (y[t] - base) % 20     # regulile care au nimerit
            mx = np.maximum(mx, run); run += 1; run[h] = 0
            hits[h] += 1; recent[h] += 1; hist.append(h)
            if len(hist) > W:
                recent[hist.pop(0)] -= 1
        t0 = t1
    return out


def report(out, title):
    print(f"\n{title}")
    print(f"{'strategie':<52}{'ture':>6}{'numere':>8}{'nimerire':>10}{'hazard':>8}{'eficiență':>11}{'z':>7}{'pauza max':>10}")
    for name, rows in out.items():
        k = np.array([len(s) for s, _ in rows]); h = np.array([r in s for s, r in rows])
        exp = k / 20
        z = (h.sum() - exp.sum()) / np.sqrt((exp * (1 - exp)).sum())
        r_ = m = 0
        for v in h:
            r_ = 0 if v else r_ + 1; m = max(m, r_)
        print(f"{name:<52}{len(h):6d}{k.mean():8.1f}{100*h.mean():9.1f}%{100*exp.mean():7.1f}%"
              f"{h.sum()/exp.sum():11.3f}{z:7.2f}{m:10d}")


# 1) jurnalul real
lg, _ = pauze.load()
P = np.array([[int(v) for v in p.split()] for p in lg.propuneri]); y = lg.rez.to_numpy(int)
report(strategies(P, y), f"JURNALUL TĂU ({len(y)} ture, strategiile joacă de la a {START}-a)")

# 2) istoricul lung, cu propunerile reconstituite
nb = json.load(open("Predictor.ipynb"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {"display": lambda *a, **k: None}
with contextlib.redirect_stdout(io.StringIO()):
    for src in code[:7]:
        exec(src, g)
T = np.arange(2000, g["N"])
Ph = np.argsort(-g["P_ens"][T], axis=1, kind="stable")[:, :9] + 1
yh = g["x"][T] + 1
report(strategies(Ph, yh, block=500), f"ISTORICUL LUNG ({len(yh):,} ture, în bucăți de 500)")
print("\n|z| < 2 = în limitele hazardului; eficiența 1,00 = exact cât dau numere alese la întâmplare.")
