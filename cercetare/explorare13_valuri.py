"""Faza 13: strategia „în valuri” — la fiecare tură alegi regula cu cele mai multe nimeriri
pe ultimele W ture și o folosești pentru tura următoare. Testat pe ~31.000 de ture.
Rulare: python3 explorare13_valuri.py <director cu Predictor.ipynb și history.csv> <predictions_log.csv>"""
import sys, os
import numpy as np
from scipy.stats import binomtest
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "explorare12_reguli_propuneri.py")).read().split("rows = [")[0]
exec(src)                                    # încarcă R_hist (1.700 de reguli × ~31.000 de ture)

names = list(R_hist)
H = np.stack([R_hist[r] for r in names], 1).astype(np.int16)        # (ture, reguli)
C = np.vstack([np.zeros((1, H.shape[1]), np.int32), np.cumsum(H, 0, dtype=np.int32)])
n = H.shape[0]
rng = np.random.default_rng(0)
print(f"{len(names):,} reguli, {n:,} ture; o regulă = un număr, șansa pură 5,00%\n")
print(f"{'fereastra (ultimele W ture)':<30} {'nimerit':>8} {'p':>6}   {'ce a mers în fereastră':>24}")
for W in (5, 10, 17, 34, 51, 119, 250, 510, 1000, 3000):
    hits, past = [], []
    for t in range(W, n):
        recent = C[t] - C[t - W]
        best = np.flatnonzero(recent == recent.max())
        j = rng.choice(best)
        hits.append(H[t, j]); past.append(recent[j] / W)
    hits = np.array(hits)
    p = binomtest(int(hits.sum()), len(hits), 0.05, alternative="greater").pvalue
    print(f"W = {W:<5} ({W/17:5.1f} zile){'':<8} {100*hits.mean():7.2f}% {p:6.2f}   {100*np.mean(past):21.1f}%")
