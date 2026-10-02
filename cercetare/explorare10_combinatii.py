"""Faza 10: căutare exhaustivă de modele și combinații de modele, evaluate după profit cu cotele reale.
Selecție pe trecut (antrenare), verificare pe ultimele 6.000 de ture (nevăzute).
Rulare: python3 explorare10_combinatii.py <director cu Predictor.ipynb și history.csv>"""
import json, os, sys, contextlib, io, itertools, time
import numpy as np, matplotlib
matplotlib.use('Agg')
from scipy.stats import binomtest

run = sys.argv[1]
nb = json.load(open(f'{run}/Predictor.ipynb'))
code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
os.chdir(run); g = {'display': lambda *a, **k: None}
with contextlib.redirect_stdout(io.StringIO()):
    for src in code[:7]: exec(src, g)
x, P, N, MODELS = g['x'], g['P'], g['N'], g['MODELS']
K20 = 20
COTE = {9: 2.11, 10: 1.90, 11: 1.73, 12: 1.58}

# ── modele de bază: cele 6 din notebook + variante ──
base = {m: P[m][:N] for m in MODELS[1:]}
onehot = np.eye(K20)[x]
for hl in (30, 100, 1000, 3000):                 # frecvențe recente cu diverse viteze de uitare
    dec = 0.5 ** (1 / hl); r = np.zeros(K20); out = np.empty((N, K20))
    for t in range(N):
        out[t] = r; r = r * dec + onehot[t]
    base[f"recentă hl={hl}"] = out
cs = np.vstack([np.zeros(K20), np.cumsum(onehot, 0)])
for W in (17, 51, 170):                          # frecvențe pe ferestre fixe
    idx = np.arange(N); lo = np.maximum(idx - W, 0)
    base[f"fereastră {W}"] = cs[idx] - cs[lo]
models = dict(base)
for m, v in base.items():                        # variantele „reci” (inversate)
    models[f"{m} (rece)"] = -v
names = list(models)

T = np.arange(2000, N); y = x[T]
tr, te = T < N - 6000, T >= N - 6000
rng = np.random.default_rng(0)
tie = rng.random((len(T), K20)) * 1e-9
Z = {}                                           # scoruri standardizate pe rând, pentru combinare
R = {}                                           # ranguri (0 = cel mai bun)
for m in names:
    s = models[m][T].astype(float)
    s = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-12)
    Z[m] = s + tie
    R[m] = np.argsort(np.argsort(-Z[m], 1), 1)

results = []   # (descriere, K, rata trecut, rata test, roi trecut, roi test)
def evaluate(desc, score):
    order = np.argsort(-score, 1)
    rank_y = np.argmax(order == y[:, None], 1)
    for K, o in COTE.items():
        h = rank_y < K
        a, b = h[tr].mean(), h[te].mean()
        results.append((desc, K, a, b, a * o - 1, b * o - 1))

t0 = time.time()
for m in names:                                  # 1) fiecare model singur
    evaluate(m, Z[m])
nb6 = MODELS[1:]                                 # 2) toate submulțimile celor 6 modele din notebook
for r_ in range(2, 7):
    for S in itertools.combinations(nb6, r_):
        lab = " + ".join(S)
        evaluate(f"[suma] {lab}", sum(Z[m] for m in S))
        evaluate(f"[rang] {lab}", -sum(R[m] for m in S))
        evaluate(f"[min-rang] {lab}", -np.min([R[m] for m in S], 0) - 1e-3 * sum(R[m] for m in S))
        evaluate(f"[vot-9] {lab}", sum((R[m] < 9) for m in S) + 1e-3 * sum(Z[m] for m in S))
for a_, b_ in itertools.combinations(names, 2):  # 3) toate perechile dintre cele 26
    evaluate(f"[suma] {a_} + {b_}", Z[a_] + Z[b_])
print(f"{len(names)} modele de bază, {len(results):,} strategii (cu tot cu 9/10/11/12 numere) în {time.time()-t0:.0f}s")
print(f"Trecut: {tr.sum():,} ture, test (nevăzut): {te.sum():,} ture\n")

import pandas as pd
D = pd.DataFrame(results, columns=["strategie", "K", "rata_trecut", "rata_test", "roi_trecut", "roi_test"])
for K, o in COTE.items():
    d = D[D.K == K]
    best = d.sort_values("roi_trecut", ascending=False).iloc[0]
    top100 = d.sort_values("roi_trecut", ascending=False).head(100)
    n_te = te.sum()
    p = binomtest(int(round(best.rata_test * n_te)), n_te, K / 20, alternative='greater').pvalue
    print(f"── {K} numere (cota {o}, randament pur {100*(o*K/20-1):+.2f}%) — {len(d):,} strategii")
    print(f"   cea mai bună pe trecut: {best.strategie}")
    print(f"      trecut: {100*best.rata_trecut:.2f}% nimerit, randament {100*best.roi_trecut:+.2f}%")
    print(f"      TEST:   {100*best.rata_test:.2f}% nimerit, randament {100*best.roi_test:+.2f}%  (p = {p:.2f})")
    print(f"   cele mai bune 100 pe trecut, în test: randament mediu {100*top100.roi_test.mean():+.2f}%")
    print(f"   toate strategiile, în test: randament mediu {100*d.roi_test.mean():+.2f}%, "
          f"cu profit în test: {100*(d.roi_test > 0).mean():.1f}%")
    print(f"   corelație trecut ↔ test: {np.corrcoef(d.roi_trecut, d.roi_test)[0,1]:+.3f}\n")

# câte strategii ar ieși „pe profit” în test din pur noroc?
K, o = 9, 2.11
sims = rng.random((2000, te.sum())) < 0.45
null_roi = sims.mean(1) * o - 1
print(f"Referință: o strategie complet aleatoare cu 9 numere iese pe profit în test în "
      f"{100*(null_roi > 0).mean():.1f}% din cazuri (doar din noroc).")
D.to_csv("/tmp/claude-0/-home-user-Predictor/9da01c27-98cd-5380-9d0a-6e35dab68510/scratchpad/combinatii.csv", index=False)
