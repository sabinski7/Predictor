"""Explorare 17: reguli „ieșit = combinație de propuneri + c” (ca p1 + p8 + 9), verificare detaliată.
Familii (numere 1..20, totul mod 20): pi + c, 2·pi + c, pi ± pj + c, −pi − pj + c, pi + pj + pk + c,
pi + pj − pk + c, suma/min/max/mediana + c  → 10.040 de reguli, fiecare prezice UN număr (șansa 5%).
Disciplina: căutare pe prima jumătate a istoricului reconstituit, verificare pe a doua jumătate
și pe jurnalul tău real (predictions_log.csv), pe care căutarea nu le-a văzut.
Rulare: python cercetare/explorare17_top_reguli_propuneri.py   (din rădăcina repo-ului)"""
import json, contextlib, io, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
from scipy.stats import binom

K = 20

def bases(P):
    """P: (n, 9) propuneri 1..20 în ordinea rangului. Întoarce (nume, vector bază) — regula e bază + c."""
    out = []
    for i in range(9):
        out.append((f"p{i+1}", P[:, i]))
        out.append((f"2·p{i+1}", 2 * P[:, i]))
    for i, j in itertools.permutations(range(9), 2):
        out.append((f"p{i+1} − p{j+1}", P[:, i] - P[:, j]))
    for i, j in itertools.combinations(range(9), 2):
        out.append((f"p{i+1} + p{j+1}", P[:, i] + P[:, j]))
        out.append((f"−p{i+1} − p{j+1}", -P[:, i] - P[:, j]))
    for i, j, k in itertools.combinations(range(9), 3):
        out.append((f"p{i+1} + p{j+1} + p{k+1}", P[:, i] + P[:, j] + P[:, k]))
    for (i, j), k in itertools.product(itertools.combinations(range(9), 2), range(9)):
        if k not in (i, j):
            out.append((f"p{i+1} + p{j+1} − p{k+1}", P[:, i] + P[:, j] - P[:, k]))
    S = np.sort(P, 1)
    out += [("suma", P.sum(1)), ("min", S[:, 0]), ("max", S[:, -1]), ("mediana", S[:, 4])]
    return out

def hit_matrix(P, y):
    """(reguli × 20 valori de c) → număr de nimeriri. Nimerire: (bază + c − 1) mod 20 + 1 == y."""
    names, cnt = [], []
    for name, b in bases(P):
        d = (y - b) % K              # c (mod 20) care ar fi nimerit
        names.append(name)
        cnt.append(np.bincount(d, minlength=K))
    return names, np.array(cnt)

def label(name, c):
    c = c if c <= 10 else c - K
    return name if c == 0 else f"{name} {'+' if c > 0 else '−'} {abs(c)}"

def predict(name_c, P1):
    """valoarea prezisă de o regulă pentru un singur rând de propuneri P1 (1..20)."""
    name, c = name_c
    b = dict(bases(P1[None, :]))[name][0]
    return int((b + c - 1) % K + 1)

# istoricul, cu propunerile notebook-ului reconstituite (top-9 ale ansamblului, fără să vadă viitorul)
nb = json.load(open("Predictor.ipynb"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {"display": lambda *a, **k: None}
with contextlib.redirect_stdout(io.StringIO()):
    for src in code[:7]:
        exec(src, g)
x, P_ens, N = g["x"], g["P_ens"], g["N"]
T = np.arange(2000, N)
P_hist = np.argsort(-P_ens[T], axis=1, kind="stable")[:, :9] + 1
y_hist = x[T] + 1
half = len(T) // 2

# jurnalul real
h = pd.read_csv("history.csv", dtype=str)
res = dict(zip(h.Date + " " + h.Time, h["Drawn Number"].astype(int)))
lg = pd.read_csv("predictions_log.csv", dtype=str)
nxt = lg[(lg.k == "9") & ~lg.slot.isin(res)].tail(1)
lg["rez"] = lg.slot.map(res)
lg = lg[(lg.k == "9") & lg.rez.notna()]
P_log = np.array([[int(v) for v in p.split()] for p in lg.propuneri])
y_log = lg.rez.astype(int).to_numpy()

names, A = hit_matrix(P_hist[:half], y_hist[:half])
_, B = hit_matrix(P_hist[half:], y_hist[half:])
_, L = hit_matrix(P_log, y_log)
nA, nB, nL = half, len(T) - half, len(y_log)
M = A.size
print(f"Istoric reconstituit: {len(T):,} ture (căutare {nA:,} + verificare {nB:,}); jurnalul tău: {nL} ture")
print(f"Reguli testate: {M:,} — fiecare prezice un număr, șansa pură 5,00%\n")

def pval(k, n):
    return binom.sf(k - 1, n, 1 / K)

# 1) top 10 la căutare și ce fac pe date nevăzute
flat = np.argsort(-A.ravel())[:10]
print("TOP 10 găsite pe prima jumătate → verificate pe a doua jumătate și pe jurnalul tău")
print(f"{'regula':<28}{'căutare':>9}{'verificare':>12}{'jurnal':>9}")
for f in flat:
    r, c = divmod(f, K)
    print(f"{label(names[r], c):<28}{100*A[r,c]/nA:8.2f}%{100*B[r,c]/nB:11.2f}%{100*L[r,c]/nL:8.2f}%")
top = A.ravel().argsort()[::-1]
for n_ in (10, 100, 1000):
    s = top[:n_]
    print(f"  media celor mai bune {n_:>4} la verificare: {100*B.ravel()[s].mean()/nB:.2f}%   (la căutare {100*A.ravel()[s].mean()/nA:.2f}%)")
print(f"  cel mai bun la căutare: p = {pval(A.max(), nA):.1e}; prag Bonferroni pentru {M:,} reguli: {0.05/M:.1e}")

# 2) top 10 pe tot istoricul (cele mai stabile), cu p corectat
Tot = A + B
flat = np.argsort(-Tot.ravel())[:10]
print("\nTOP 10 pe tot istoricul (31.000+ ture), cu jurnalul și semnificația")
print(f"{'regula':<28}{'istoric':>9}{'jumătatea 1':>13}{'jumătatea 2':>13}{'jurnal':>9}{'p corectat':>12}")
top10 = []
for f in flat:
    r, c = divmod(f, K)
    pc = min(1.0, pval(Tot[r, c], nA + nB) * M)
    top10.append((names[r], c))
    print(f"{label(names[r], c):<28}{100*Tot[r,c]/(nA+nB):8.2f}%{100*A[r,c]/nA:12.2f}%{100*B[r,c]/nB:12.2f}%"
          f"{100*L[r,c]/nL:8.2f}%{pc:12.2f}")

# 3) top 10 în jurnal (cum era p1 + p8 + 9) și pe istoric
flat = np.argsort(-L.ravel())[:10]
print(f"\nTOP 10 în jurnalul tău ({nL} ture) și cum se comportă pe istoric")
print(f"{'regula':<28}{'jurnal':>9}{'istoric':>10}")
for f in flat:
    r, c = divmod(f, K)
    print(f"{label(names[r], c):<28}{100*L[r,c]/nL:8.2f}%{100*Tot[r,c]/(nA+nB):9.2f}%")
i = names.index("p1 + p8")
print(f"  p1 + p8 + 9: jurnal {100*L[i,9]/nL:.2f}%, istoric {100*Tot[i,9]/(nA+nB):.2f}%")
print(f"  regula cea mai bună în jurnal: p = {pval(L.max(), nL):.1e}, șansa ca măcar una din {M:,} să ajungă acolo din noroc: "
      f"{100*min(1, pval(L.max(), nL)*M):.0f}%")
print(f"  corelație jurnal ↔ istoric între reguli: {np.corrcoef(L.ravel(), Tot.ravel())[0,1]:+.3f}")

# 4) ce număr dă fiecare regulă din top-ul pe istoric pentru următoarea tură
if len(nxt):
    P1 = np.array([int(v) for v in nxt.propuneri.iloc[0].split()])
    print(f"\nPentru {nxt.slot.iloc[0]} (propuneri {' '.join(map(str, P1))}), top 10 pe istoric dau:")
    print("  " + ", ".join(f"{label(n, c)} → {predict((n, c), P1)}" for n, c in top10))

# 5) control: cât de sus ajunge cea mai bună regulă dacă numerele ieșite sunt amestecate (fără nicio legătură)
rng = np.random.default_rng(0)
mx = []
for _ in range(20):
    _, Z = hit_matrix(P_hist, rng.permutation(y_hist))
    mx.append(Z.max() / len(T))
print(f"\nControl, 20 de amestecări: cea mai bună regulă pe tot istoricul ajunge la {100*np.mean(mx):.2f}% "
      f"(între {100*min(mx):.2f}% și {100*max(mx):.2f}%), real: {100*Tot.max()/(nA+nB):.2f}%")
