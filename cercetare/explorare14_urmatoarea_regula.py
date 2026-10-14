"""Faza 14: variante ale strategiei „în valuri”, pe 1.700 de reguli și ~31.000 de ture.
 A) regula „care urmează”: dacă acum cea mai bună (pe ultimele W ture) e regula R1, joci regula care,
    în trecut, a nimerit cel mai des la tura de după momentele în care R1 era cea mai bună
    (tabel învățat din mers, doar din trecut);
 B) regula cea mai slabă pe ultimele W ture („îi vine rândul”).
Rulare: python3 explorare14_urmatoarea_regula.py <director cu Predictor.ipynb și history.csv> <predictions_log.csv>"""
import os, sys
import numpy as np
from scipy.stats import binomtest
__file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), "explorare13_valuri.py")).read().split('print(f"{len(names)')[0])

R = H.shape[1]
print(f"{R:,} reguli, {n:,} ture; o regulă = un număr, șansa pură 5,00%\n")
print(f"{'fereastra':<22} {'A) regula care urmează':>24} {'B) cea mai slabă':>18}")
for W in (17, 51, 119, 510):
    rng = np.random.default_rng(1)
    M = np.zeros((R, R), np.int32)          # M[r1, r2] = de câte ori a nimerit r2 imediat după ce r1 era cea mai bună
    seen = np.zeros(R, np.int32)
    hA, hB = [], []
    for t in range(W, n):
        recent = C[t] - C[t - W]
        r1 = rng.choice(np.flatnonzero(recent == recent.max()))
        # A
        if seen[r1] >= 20:
            row = M[r1]
            rA = rng.choice(np.flatnonzero(row == row.max()))
            hA.append(H[t, rA])
        # B
        rB = rng.choice(np.flatnonzero(recent == recent.min()))
        hB.append(H[t, rB])
        M[r1] += H[t]; seen[r1] += 1         # învățare după ce tura s-a jucat
    hA, hB = np.array(hA), np.array(hB)
    pA = binomtest(int(hA.sum()), len(hA), .05, alternative="greater").pvalue if len(hA) else float("nan")
    pB = binomtest(int(hB.sum()), len(hB), .05, alternative="greater").pvalue
    print(f"W = {W:<4} ({W/17:4.1f} zile)   {100*hA.mean():6.2f}% (p={pA:.2f}, {len(hA):,} ture)   {100*hB.mean():6.2f}% (p={pB:.2f})")
