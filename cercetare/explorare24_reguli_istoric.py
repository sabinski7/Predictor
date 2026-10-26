"""Explorare 24: reguli de același tip ca „p3 + p4 − p2 − 3”, dar pe numerele ieșite anterior, nu pe propuneri.
Intrări: n1 = numărul ieșit acum 1 tură, n2 = acum 2 ture, …, n9 (ultimele 9 extrageri), ca p1…p9 la propuneri.
Reguli: combinații cu semne ± de 1–4 numere, ±2·ni, plus o constantă (mod 20) → 57.360 de reguli, ca la propuneri.
Verificări:
  1) cele mai bune reguli pe prima jumătate a istoricului → cum merg pe a doua jumătate;
  2) strategia „top 10 cu cele mai scurte pauze” (ca mesajul 3) pe tot istoricul, în bucăți de 500 de ture;
  3) aceeași strategie pe perioada jurnalului tău, comparată cu varianta pe propuneri.
Rulare din rădăcina repo-ului: python cercetare/explorare24_reguli_istoric.py"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "automat")
import pauze

C = pauze.C; NB = len(C); NR = NB * 20
h = pd.read_csv("history.csv")
x = h["Drawn Number"].to_numpy(); slots = (h.Date + " " + h.Time).to_numpy(); N = len(x)
LAG = np.stack([np.r_[np.zeros(k, int), x[:-k]] for k in range(1, 10)], 1)   # LAG[t] = (n1..n9)


def label(f):
    return pauze.name(f).replace("p", "n")


# 1) căutare pe jumătatea 1, verificare pe jumătatea 2 (rata fiecărei reguli)
t = np.arange(9, N); half = 9 + (N - 9) // 2
cntA = np.zeros(NR, np.int64); cntB = np.zeros(NR, np.int64)
for i in t:
    hit = np.arange(NB) * 20 + (x[i] - LAG[i] @ C.T) % 20
    (cntA if i < half else cntB)[hit] += 1
nA, nB = half - 9, N - half
best = np.argsort(-cntA)[:10]
print(f"Istoric: {N:,} extrageri; {NR:,} reguli pe ultimele 9 numere ieșite; șansa pură 5%\n")
print("1) Cele mai bune 10 reguli pe prima jumătate → a doua jumătate")
for f in best:
    print(f"   {label(f):<28} {100*cntA[f]/nA:5.2f}% → {100*cntB[f]/nB:5.2f}%")
top = np.argsort(-cntA)
for k in (10, 100, 1000):
    print(f"   media primelor {k:>4}: {100*cntA[top[:k]].mean()/nA:.2f}% → {100*cntB[top[:k]].mean()/nB:.2f}%")
print(f"   corelație jumătatea 1 ↔ 2 între reguli: {np.corrcoef(cntA, cntB)[0,1]:+.3f}")


# 2) + 3) strategia top 10 pauze scurte (ca la propuneri)
def top10_strategy(inputs, y, block=500, start=50):
    """întoarce, pentru fiecare tură jucată: (numerele celor 10 reguli, numărul ieșit, numărul regulii de pe locul 1)."""
    out = []
    n = len(y)
    for b0 in range(0, n, block or n):
        b1 = min(n, b0 + (block or n))
        run = np.zeros(NR, np.int32); mx = run.copy(); hits = run.copy()
        for i in range(b0, b1):
            base = inputs[i] @ C.T
            if i - b0 >= start:
                key = np.maximum(mx, run).astype(np.int64) * 100000 - hits
                ii = np.argpartition(key, 10)[:10]; ii = ii[np.lexsort((ii, key[ii]))]
                pred = [int((base[f // 20] + f % 20 - 1) % 20 + 1) for f in ii]
                out.append((pred, y[i]))
            hit = np.arange(NB) * 20 + (y[i] - base) % 20
            mx = np.maximum(mx, run); run += 1; run[hit] = 0; hits[hit] += 1
    return out


def summary(res, name):
    allr = np.mean([np.mean([p == y for p in pr]) for pr, y in res])
    loc1 = np.mean([pr[0] == y for pr, y in res])
    anyh = np.array([y in set(pr) for pr, y in res]); k = np.mean([len(set(pr)) for pr, _ in res])
    r_ = m = 0
    for v in anyh:
        r_ = 0 if v else r_ + 1; m = max(m, r_)
    print(f"   {name:<44} ture {len(res):>6,}  fiecare regulă {100*allr:5.2f}%  locul 1 {100*loc1:5.2f}%  "
          f"măcar una {100*anyh.mean():5.1f}% (hazard {100*k/20:4.1f}%)  pauza max {m}")


print("\n2) Strategia „top 10 pauze scurte” pe tot istoricul (bucăți de 500 de ture)")
summary(top10_strategy(LAG[9:], x[9:]), "reguli pe numerele ieșite")

print("\n3) Pe perioada jurnalului tău, aceeași strategie")
lg, _ = pauze.load()
pos = {s: i for i, s in enumerate(slots)}
ii = np.array([pos[s] for s in lg.slot])
y = lg.rez.to_numpy(int)
P = np.array([[int(v) for v in p.split()] for p in lg.propuneri])
summary(top10_strategy(LAG[ii], y, block=None), "reguli pe numerele ieșite (n1…n9)")
summary(top10_strategy(P, y, block=None), "reguli pe propuneri (p1…p9, mesajul 3)")
