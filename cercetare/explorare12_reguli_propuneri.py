"""Faza 12: există o relație aritmetică între propuneri și numărul ieșit?
(ex. ieșit = primul propus − 1, = p3 + 5, = p1 + p2, = suma propunerilor ...)
Regulile se caută în jurnalul real (predictions_log.csv) și se verifică pe ~31.000 de ture din istoric,
cu propunerile reconstituite ale notebook-ului (top-9 ale ansamblului, fără să vadă viitorul).
Rulare: python3 explorare12_reguli_propuneri.py <director cu Predictor.ipynb, history.csv> <predictions_log.csv>"""
import json, os, sys, contextlib, io, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
from scipy.stats import binom

run, log_path = sys.argv[1], os.path.abspath(sys.argv[2])
K = 20

def rules_hits(P, y):
    """P: (n, 9) propuneri 0..19 în ordinea rangului; y: (n,) numărul ieșit 0..19.
    Întoarce dict regulă -> vector bool de nimeriri (fiecare regulă prezice UN număr)."""
    out = {}
    def fam(name, base):
        d = (y - base) % K                      # c-ul care ar fi nimerit, pentru fiecare tură
        for c in range(K):
            out[f"{name} {'+' if c <= 10 else '-'}{c if c <= 10 else K - c}"] = d == c
    for i in range(9):
        fam(f"p{i+1}", P[:, i])
    for i, j in itertools.combinations(range(9), 2):
        fam(f"p{i+1}+p{j+1}", P[:, i] + P[:, j] + 1)        # +1: lucrăm cu numere 1..20
        fam(f"p{i+1}-p{j+1}", P[:, i] - P[:, j])
    S = P.sum(1) + 9
    fam("suma", S - 1)
    fam("min", P.min(1)); fam("max", P.max(1)); fam("mediana", np.sort(P, 1)[:, 4])
    return out

# 1) jurnalul real
h = pd.read_csv(os.path.join(run, "history.csv"), dtype=str)
res = dict(zip(h.Date + " " + h.Time, h["Drawn Number"].astype(int)))
lg = pd.read_csv(log_path, dtype=str)
lg["rez"] = lg.slot.map(res)
lg = lg[(lg.k == "9") & lg.rez.notna()]
P_log = np.array([[int(v) - 1 for v in p.split()] for p in lg.propuneri])
y_log = lg.rez.astype(int).to_numpy() - 1
R_log = rules_hits(P_log, y_log)
n_log = len(y_log)

# 2) istoricul, cu propunerile reconstituite ale notebook-ului
nb = json.load(open(os.path.join(run, "Predictor.ipynb")))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
os.chdir(run); g = {"display": lambda *a, **k: None}
with contextlib.redirect_stdout(io.StringIO()):
    for src in code[:7]:
        exec(src, g)
x, P_ens, N = g["x"], g["P_ens"], g["N"]
T = np.arange(2000, N)
P_hist = np.argsort(-P_ens[T], axis=1, kind="stable")[:, :9]
y_hist = x[T]
R_hist = rules_hits(P_hist, y_hist)

rows = [(r, h_.mean(), R_hist[r].mean()) for r, h_ in R_log.items()]
D = pd.DataFrame(rows, columns=["regula", "jurnal", "istoric"]).sort_values("jurnal", ascending=False)
best = D.iloc[0]
k_best = int(round(best.jurnal * n_log))
p_any = 1 - binom.cdf(k_best - 1, n_log, 1 / K) ** 0 * 0  # placeholder (calculat mai jos)
p_one = binom.sf(k_best - 1, n_log, 1 / K)
p_any = 1 - (1 - p_one) ** len(D)
print(f"Jurnal: {n_log} ture cu 9 propuneri. Istoric reconstituit: {len(y_hist):,} ture.")
print(f"Reguli testate: {len(D):,} (fiecare prezice un singur număr → șansa pură 5,00%)\n")
print("Cele mai bune 10 reguli în jurnalul tău și cum se comportă pe istoric:")
for _, r in D.head(10).iterrows():
    print(f"  {r.regula:<16} jurnal {100*r.jurnal:5.2f}%   istoric {100*r.istoric:5.2f}%")
print(f"\nCea mai bună regulă din jurnal: {best.regula} → {100*best.jurnal:.1f}% ({k_best}/{n_log}).")
print(f"Șansa ca, din {len(D):,} reguli fără sens, măcar una să ajungă aici din noroc: {100*p_any:.0f}%")
print(f"Pe istoric, aceeași regulă: {100*best.istoric:.2f}%")
print(f"Media pe istoric a celor mai bune 50 de reguli din jurnal: {100*D.head(50).istoric.mean():.2f}%")
print(f"Cea mai bună regulă pe istoric (oricare): {D.sort_values('istoric').iloc[-1].regula} → {100*D.istoric.max():.2f}%")
print(f"Corelație jurnal ↔ istoric între reguli: {np.corrcoef(D.jurnal, D.istoric)[0,1]:+.3f}")
for q in ["p1 -1", "p1 +1", "p1 +0"]:
    r = D[D.regula == q].iloc[0]
    print(f"  exemplu „{q}”: jurnal {100*r.jurnal:.2f}%, istoric {100*r.istoric:.2f}%")
