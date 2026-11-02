"""Explorare 27: cele 11 numere „prezise de cele mai multe reguli”, din toate cele 1,4 milioane de reguli pe istoric.
1) Vot simplu (fiecare regulă = un vot): fiecare formulă apare cu toate cele 20 de constante, deci fiecare număr
   primește exact același număr de voturi — votul simplu nu poate alege nimic.
2) Vot ponderat: fiecare regulă votează cu numărul ei de nimeriri pe ultimele 600 de ture (regulile care „merg” contează mai mult).
3) Vot doar al celor mai bune 1.000 de reguli după pauza maximă pe ultimele 600 de ture.
Testat tură cu tură, doar cu extragerile de dinainte; comparat cu 11 numere la întâmplare (55%).
Rulare: python cercetare/explorare27_vot_toate_regulile.py <ture_vot_ponderat> <ture_top1000>"""
import sys
import numpy as np
sys.path.insert(0, "automat")
import istoric_reguli as R

slots, x, _ = R._load()
n = len(x); W = R.WINDOW
CT = R.CT
n_w = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
n_t = int(sys.argv[2]) if len(sys.argv) > 2 else 300

# 1) vot simplu
t = n
inp = R.inputs(x, t)
base = inp @ CT
votes = np.bincount(((base[:, None] + np.arange(20) - 1) % 20).ravel(), minlength=20)
print(f"1) Vot simplu, toate {R.NB*20:,} regulile: voturi pe număr min {votes.min():,}, max {votes.max():,} → toate egale\n")


def maxpause(h):
    r = m = 0
    for v in h:
        r = 0 if v else r + 1; m = max(m, r)
    return m


# 2) vot ponderat cu nimeririle pe ultimele 600 de ture
hits = []
LAGX = np.stack([np.r_[np.zeros(L, int), x[:-L]] for L in R.LAGS], 1)     # LAGX[u] = intrările turei u
BASE = None
for t in range(n - n_w, n):
    lo = t - W
    Bw = LAGX[lo:t] @ CT                     # (600, NB) bazele în fereastră
    bn = LAGX[t] @ CT                        # (NB,) bazele acum
    pred_if = (x[lo:t, None] - Bw + bn[None, :]) % 20      # numărul pe care l-ar prezice acum fiecare regulă care a nimerit la u
    v = np.bincount(pred_if.ravel(), minlength=20)         # index 0..19 ↔ numerele 1..20 (mod 20: 0 ↔ 20)
    num = np.where(np.arange(20) == 0, 20, np.arange(20))
    top11 = num[np.argsort(-v, kind="stable")[:11]]
    hits.append(x[t] in top11)
h = np.array(hits)
print(f"2) Vot ponderat cu nimeririle (ultimele {n_w} ture): {100*h.mean():.2f}% (hazard 55%), pauza max {maxpause(h)}")

# 3) doar cele mai bune 1.000 de reguli după pauză
hits = []
for t in range(n - n_t, n):
    key, _, _ = R.keys(x, t)
    best = np.argpartition(key, 1000)[:1000]
    inp = R.inputs(x, t)
    preds = [R.predict(f, inp) for f in best]
    v = np.bincount(preds, minlength=21)[1:]
    top11 = np.argsort(-v, kind="stable")[:11] + 1
    hits.append(x[t] in top11)
h = np.array(hits)
print(f"3) Vot al celor mai bune 1.000 de reguli (ultimele {n_t} ture): {100*h.mean():.2f}% (hazard 55%), pauza max {maxpause(h)}")
rng = np.random.default_rng(0)
r = np.array([x[t] in rng.choice(20, 11, replace=False) + 1 for t in range(n - n_w, n)])
print(f"\nControl, 11 numere la întâmplare pe ultimele {n_w} ture: {100*r.mean():.2f}%, pauza max {maxpause(r)}")
