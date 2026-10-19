"""Explorare 18: top 10 reguli din propuneri cu cea mai scurtă pauză maximă între nimeriri, pe jurnalul tău.
Reguli: orice combinație cu semne ± de 1–4 propuneri, sau ±2·pi, plus orice constantă (mod 20) → 57.360 de reguli.
Pauza maximă include și turele de la începutul jurnalului până la prima nimerire și pauza în curs.
La egalitate câștigă regula cu mai multe nimeriri. Doar turele cu 9 propuneri și rezultat cunoscut.
Rulare: python cercetare/explorare18_pauze_scurte.py   (din rădăcina repo-ului)"""
import itertools, numpy as np, pandas as pd

h = pd.read_csv("history.csv", dtype=str)
res = dict(zip(h.Date + " " + h.Time, h["Drawn Number"].astype(int)))
lg = pd.read_csv("predictions_log.csv", dtype=str)
nxt = lg[(lg.k == "9") & ~lg.slot.isin(res)].tail(1)
lg["rez"] = lg.slot.map(res)
lg = lg[(lg.k == "9") & lg.rez.notna()]
P = np.array([[int(v) for v in p.split()] for p in lg.propuneri])
y = lg.rez.astype(int).to_numpy()
n = len(y)

rules = []                                   # (nume, coeficienți pe cele 9 poziții)
for r in range(1, 5):
    for S in itertools.combinations(range(9), r):
        for sg in itertools.product([1, -1], repeat=r):
            co = np.zeros(9, int); co[list(S)] = sg
            rules.append(co)
for i in range(9):
    for s in (2, -2):
        co = np.zeros(9, int); co[i] = s
        rules.append(co)
C = np.array(rules)
base = P @ C.T                               # (n, reguli)

def name(co, c):
    parts = [f"{'' if k == 2 or k == 1 else '−'}{'2·' if abs(k) == 2 else ''}p{i+1}" for i, k in enumerate(co) if k > 0]
    parts += [f"− {'2·' if k == -2 else ''}p{i+1}" for i, k in enumerate(co) if k < 0]
    s = " + ".join(p for p in parts if not p.startswith("− ")) or ""
    neg = " ".join(p for p in parts if p.startswith("− "))
    s = (s + " " + neg).strip() if s else "−" + neg[2:].replace(" − ", " − ")
    c = c if c <= 10 else c - 20
    return s if c == 0 else f"{s} {'+' if c > 0 else '−'} {abs(c)}"

H = ((y[:, None] - base) % 20)[:, :, None] == np.arange(20)      # (n, reguli, c)
H = H.reshape(n, -1)
run = np.zeros(H.shape[1], int); mx = np.zeros(H.shape[1], int)
for t in range(n):
    mx = np.maximum(mx, np.where(H[t], run, 0))
    run = np.where(H[t], 0, run + 1)
mx = np.maximum(mx, run)
hits = H.sum(0)
order = np.lexsort((-hits, mx))[:10]

P1 = np.array([int(v) for v in nxt.propuneri.iloc[0].split()]) if len(nxt) else None
print(f"Jurnal: {n} ture cu 9 propuneri ({lg.slot.iloc[0]} – {lg.slot.iloc[-1]}), {H.shape[1]:,} reguli verificate")
if P1 is not None:
    print(f"Următoarea tură: {nxt.slot.iloc[0]}, propuneri {' '.join(map(str, P1))}")
print(f"\n{'#':>2}  {'regula':<30}{'pauza max':>10}{'nimeriri':>12}{'acum fără':>11}{'prezice':>9}")
for k, f in enumerate(order, 1):
    r, c = divmod(f, 20)
    pr = int((P1 @ C[r] + c - 1) % 20 + 1) if P1 is not None else ""
    print(f"{k:>2}  {name(C[r], c):<30}{mx[f]:>10}{hits[f]:>6} ({100*hits[f]/n:.1f}%){run[f]:>9}{pr:>9}")
