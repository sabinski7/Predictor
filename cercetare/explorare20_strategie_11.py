"""Explorare 20: strategia cu cea mai mică pauză între nimeriri, cu cel mult 11 numere.
Compară, pe același istoric lung (31.000+ ture), mai multe feluri de a alege 11 numere:
regulile din top 10 + propunerile notebook-ului, primele 11 ale ansamblului, cele mai „reci”, cele mai „calde”,
11 numere fixe, 11 numere la întâmplare. Pauza = ture la rând fără nimerire.
Rulare din rădăcina repo-ului: python cercetare/explorare20_strategie_11.py"""
import sys, json, contextlib, io
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "automat")
import pauze

C = pauze.C; NB = len(C); NR = NB * 20; K = 11

nb = json.load(open("Predictor.ipynb"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {"display": lambda *a, **k: None}
with contextlib.redirect_stdout(io.StringIO()):
    for src in code[:7]:
        exec(src, g)
x, P_ens, N = g["x"] + 1, g["P_ens"], g["N"]
T0 = 2000
order = np.argsort(-P_ens, axis=1, kind="stable") + 1         # numerele ansamblului, în ordinea probabilității


def fill(nums, t):
    """completează până la 11 cu numerele cele mai frecvente în ultimele 100 de extrageri."""
    nums = list(dict.fromkeys(nums))[:K]
    if len(nums) < K:
        freq = np.bincount(x[max(0, t - 100):t], minlength=21)[1:]
        for v in np.argsort(-freq, kind="stable") + 1:
            if v not in nums:
                nums.append(int(v))
            if len(nums) == K:
                break
    return set(nums)


rng = np.random.default_rng(0)
S = {k: [] for k in ["A. reguli top 10 + propuneri (+ completare)", "B. primele 11 ale ansamblului",
                     "C. cele 11 cele mai „reci” (neieșite de mult)", "D. cele 11 cele mai „calde” (ultimele 100)",
                     "E. 11 numere fixe (1–11)", "F. 11 numere la întâmplare"]}
last_seen = np.zeros(21, int)
for t in range(T0):
    last_seen[x[t]] = t
block = 500
for b0 in range(T0, N, block):
    run = np.zeros(NR, np.int32); mx = np.zeros(NR, np.int32); hits = np.zeros(NR, np.int32)
    for t in range(b0, min(N, b0 + block)):
        props = order[t, :9]
        base = props @ C.T
        if t - b0 >= 50:
            key = np.maximum(mx, run).astype(np.int64) * 100000 - hits
            idx = np.argpartition(key, 10)[:10]; idx = idx[np.argsort(key[idx], kind="stable")]
            rules = [int((base[i // 20] + i % 20 - 1) % 20 + 1) for i in idx]
            y = x[t]
            S["A. reguli top 10 + propuneri (+ completare)"].append(y in fill(rules + list(props), t))
            S["B. primele 11 ale ansamblului"].append(y in set(order[t, :K]))
            S["C. cele 11 cele mai „reci” (neieșite de mult)"].append(y in set(np.argsort(last_seen[1:], kind="stable")[:K] + 1))
            freq = np.bincount(x[t - 100:t], minlength=21)[1:]
            S["D. cele 11 cele mai „calde” (ultimele 100)"].append(y in set(np.argsort(-freq, kind="stable")[:K] + 1))
            S["E. 11 numere fixe (1–11)"].append(y <= K)
            S["F. 11 numere la întâmplare"].append(y in set(rng.choice(20, K, replace=False) + 1))
        h = np.arange(NB) * 20 + (x[t] - base) % 20
        mx = np.maximum(mx, run); run += 1; run[h] = 0; hits[h] += 1
        last_seen[x[t]] = t


def pauses(h):
    out, r = [], 0
    for v in h:
        if v:
            out.append(r); r = 0
        else:
            r += 1
    out.append(r)
    return np.array(out)


n = len(next(iter(S.values())))
print(f"{n:,} ture din istoric; șansa de nimerire cu 11 numere: 55%\n")
print(f"{'strategie':<48}{'nimerire':>9}{'pauza max':>10}{'pauze 5+':>9}{'pauze 7+':>9}{'pauze 9+':>9}")
for name, h in S.items():
    h = np.array(h); p = pauses(h)
    print(f"{name:<48}{100*h.mean():8.1f}%{p.max():10d}{(p>=5).sum():9d}{(p>=7).sum():9d}{(p>=9).sum():9d}")
q = 1 - K / 20
nh = n * K / 20
print(f"\nTeorie (extragere aleatoare, 11 numere): pauze 5+ ≈ {nh*q**5:.0f}, 7+ ≈ {nh*q**7:.0f}, 9+ ≈ {nh*q**9:.0f}; "
      f"pauza max tipică ≈ {np.log(nh)/np.log(1/q):.0f}")
print("Șansa ca următoarele k ture să fie toate ratate, din orice moment: " + ", ".join(f"{k}: {100*q**k:.1f}%" for k in range(1, 9)))
