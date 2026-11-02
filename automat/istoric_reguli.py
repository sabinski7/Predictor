"""Reguli doar pe istoric (fără propuneri), recalculate după fiecare tură, pe ultimele 600 de extrageri.

Reguli: „n2 + n5 − n17 + 3” etc. — nL = numărul ieșit acum L ture (L = 1…17, 34, 51), combinații cu semne ±
de 1–4 numere sau ±2·nL, plus o constantă (mod 20) → 1.410.560 de reguli (aceleași ca în cercetare/explorare26).
Clasament: cea mai scurtă pauză maximă pe ultimele 600 de ture (inclusiv pauza în curs), la egalitate mai multe nimeriri.

Scrie (test pe date noi, fără mesaje pe Telegram):
  istoric_top.csv        — locurile 1–20 la fiecare tură: regula, numărul prezis, pauza max, nimeriri, dacă e nouă
                           în top 10 și pe ce loc era la tura precedentă („de unde vine”), rezultatul;
  strategie_istoric.csv  — trei strategii, tură cu tură:
                           A = numerele regulilor de pe locurile 1–10, completate până la 11 numere;
                           B = cele 9 propuneri ale notebook-ului, completate până la 11 (comparație simplă);
                           C = numerele regulilor de pe locurile 11–20 („de la mijloc”).
Completarea până la 11: cele mai frecvente numere din ultimele 100 de extrageri.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOP_CSV = ROOT / "istoric_top.csv"
STRAT_CSV = ROOT / "strategie_istoric.csv"
LAGS = list(range(1, 18)) + [34, 51]
WINDOW = 600
K11 = 11


def _coefs():
    out = []
    for r in range(1, 5):
        for S in itertools.combinations(range(len(LAGS)), r):
            for sg in itertools.product([1, -1], repeat=r):
                co = np.zeros(len(LAGS), np.int64); co[list(S)] = sg; out.append(co)
    for i in range(len(LAGS)):
        for s in (2, -2):
            co = np.zeros(len(LAGS), np.int64); co[i] = s; out.append(co)
    return np.array(out)


C = _coefs()
NB = len(C)
CT = C.T.copy()


def name(f):
    co, c = C[f // 20], f % 20
    lab = lambda i, k: f"{'2·' if abs(k) == 2 else ''}n{LAGS[i]}"
    pos = [lab(i, k) for i, k in enumerate(co) if k > 0]
    neg = [lab(i, k) for i, k in enumerate(co) if k < 0]
    s = " + ".join(pos) + "".join(f" − {p}" for p in neg) if pos else "−" + " − ".join(neg)
    c = c if c <= 10 else c - 20
    return s if c == 0 else f"{s} {'+' if c > 0 else '−'} {abs(c)}"


def inputs(x, t):
    """(n1, n2, …) pentru tura t (indice în x)."""
    return np.array([x[t - L] for L in LAGS])


def keys(x, t):
    """Cheia de clasament a fiecărei reguli pentru tura t, din turele [t−600, t). Întoarce (cheie, pauza max, nimeriri)."""
    lo = t - WINDOW
    X = np.stack([x[lo - L:t - L] for L in LAGS], 1)
    y = x[lo:t]
    mx = np.full((NB, 20), WINDOW, np.int64); cnt = np.zeros((NB, 20), np.int64)
    CH = 4000
    for b0 in range(0, NB, CH):
        D = ((y[:, None] - X @ CT[:, b0:b0 + CH]) % 20).astype(np.int8)
        m, c = _gap_stats(D)
        mx[b0:b0 + CH] = m; cnt[b0:b0 + CH] = c
    mx, cnt = mx.ravel(), cnt.ravel()
    return mx * 1_000_000 - cnt, mx, cnt


def _gap_stats(D):
    n, nb = D.shape
    order = np.argsort(D, axis=0, kind="stable")
    Ds = np.take_along_axis(D, order, 0); tm = order
    first = np.ones((n, nb), bool); first[1:] = Ds[1:] != Ds[:-1]
    last = np.ones((n, nb), bool); last[:-1] = first[1:]
    prev = np.empty((n, nb), np.int64); prev[0] = tm[0]; prev[1:] = tm[1:] - tm[:-1] - 1
    val = np.where(first, tm, prev)
    val = np.where(last, np.maximum(val, n - 1 - tm), val)
    mx = np.full((nb, 20), n, np.int64); cnt = np.zeros((nb, 20), np.int64)
    valT, firstT, DsT = val.T.ravel(), first.T.ravel(), Ds.T.ravel()
    starts = np.flatnonzero(firstT)
    col = starts // n; cc = DsT[starts]
    mx[col, cc] = np.maximum.reduceat(valT, starts)
    cnt[col, cc] = np.diff(np.r_[starts, len(valT)])
    return mx, cnt


def ranking(key, k=20):
    ii = np.argpartition(key, k)[:k]
    return ii[np.lexsort((ii, key[ii]))]


def rank_of(key, f):
    """locul regulii f în clasamentul complet (1 = primul)."""
    return int((key < key[f]).sum() + ((key == key[f]) & (np.arange(len(key)) < f)).sum()) + 1


def predict(f, inp):
    return int((inp @ C[f // 20] + f % 20 - 1) % 20 + 1)


def fill11(nums, x_before):
    nums = list(dict.fromkeys(int(v) for v in nums))[:K11]
    if len(nums) < K11:
        freq = np.bincount(np.asarray(x_before[-100:], int), minlength=21)[1:]
        nums += [int(v) for v in np.argsort(-freq, kind="stable") + 1 if v not in nums][:K11 - len(nums)]
    return nums


def _load():
    h = pd.read_csv(ROOT / "history.csv", dtype=str)
    slots = (h.Date + " " + h.Time).tolist()
    x = h["Drawn Number"].astype(int).to_numpy()
    lg = pd.read_csv(ROOT / "predictions_log.csv", dtype=str)
    lg = lg[lg.k == "9"].drop_duplicates("slot", keep="last").set_index("slot").propuneri
    return slots, x, lg


def _next_slot(last):
    import datetime as dt
    d = dt.datetime.strptime(last, "%Y-%m-%d %H:%M")
    d = d + dt.timedelta(hours=1) if d.hour < 23 else (d + dt.timedelta(days=1)).replace(hour=7)
    return d.strftime("%Y-%m-%d %H:%M")


def update(backfill=0, start_live=None):
    """Adaugă rândurile lipsă (inclusiv tura următoare) și completează rezultatele. backfill = câte ture din trecut
    să calculeze la prima rulare (marcate test_live = nu). Întoarce câte ture noi au fost scrise."""
    slots, x, lg = _load()
    n = len(x)
    top = pd.read_csv(TOP_CSV, dtype=str) if TOP_CSV.exists() else None
    done = set(top.slot) if top is not None else set()
    first = n - backfill if top is None else slots.index(top.slot.max()) + 1 if top.slot.max() in slots else n
    targets = list(range(max(first, 51 + WINDOW), n + 1))          # n = tura următoare (fără rezultat încă)
    targets = [t for t in targets if (slots[t] if t < n else _next_slot(slots[-1])) not in done]
    new_top, prev_key = [], None
    for t in targets:
        slot = slots[t] if t < n else _next_slot(slots[-1])
        key, mx, cnt = keys(x, t)
        if prev_key is None:
            prev_key = keys(x, t - 1)[0]
        order = ranking(key)
        prev_top10 = set(ranking(prev_key)[:10])
        inp = inputs(x, t)
        res = int(x[t]) if t < n else None
        live = "da" if top is not None or backfill == 0 or t >= n else "nu"
        for loc, f in enumerate(order, 1):
            p = predict(f, inp)
            nou = loc <= 10 and f not in prev_top10
            new_top.append([slot, loc, name(f), p, mx[f], cnt[f], "da" if nou else "",
                            rank_of(prev_key, f) if nou else "", "" if res is None else res,
                            "" if res is None else ("da" if p == res else "nu"), live])
        prev_key = key
    cols = ["slot", "loc", "regula", "prezis", "pauza_max", "nimeriri", "nou_in_top10", "loc_anterior",
            "iesit", "nimerit", "test_live"]
    out = pd.DataFrame(new_top, columns=cols).astype(str)
    if top is not None:
        res = dict(zip(slots, x.astype(str)))
        pend = (top.iesit.fillna("") == "") & top.slot.isin(res)
        top.loc[pend, "iesit"] = top.loc[pend, "slot"].map(res)
        top.loc[pend, "nimerit"] = np.where(top.loc[pend, "prezis"] == top.loc[pend, "iesit"], "da", "nu")
        out = pd.concat([top, out], ignore_index=True)
    out = out.sort_values(["slot", "loc"], key=lambda s: s.astype(int) if s.name == "loc" else s, kind="stable")
    out.to_csv(TOP_CSV, index=False)
    _strategies(out, slots, x, lg)
    return len(targets)


def _strategies(top, slots, x, lg):
    pos = {s: i for i, s in enumerate(slots)}
    rows, run = [], {"A": 0, "B": 0, "C": 0}
    for slot, g in top.groupby("slot", sort=True):
        g = g.sort_values("loc", key=lambda s: s.astype(int))
        before = x[:pos.get(slot, len(x))]
        a = fill11(g.prezis.astype(int).iloc[:10], before)
        props = [int(v) for v in lg[slot].split()] if slot in lg.index else []
        b = fill11(props, before) if props else []
        c = list(dict.fromkeys(g.prezis.astype(int).iloc[10:20]))
        res = g.iesit.iloc[0]
        row = [slot, g.test_live.iloc[0], " ".join(map(str, a)), " ".join(map(str, b)), " ".join(map(str, c)), res]
        for k, nums in (("A", a), ("B", b), ("C", c)):
            if isinstance(res, str) and res and nums:
                hit = int(res) in nums
                run[k] = 0 if hit else run[k] + 1
                row += ["da" if hit else "nu", run[k]]
            else:
                row += ["", run[k]]
        rows.append(row)
    pd.DataFrame(rows, columns=["slot", "test_live", "A_numere_reguli_1_10", "B_numere_propuneri", "C_numere_reguli_11_20",
                                "iesit", "A_nimerit", "A_pauza", "B_nimerit", "B_pauza", "C_nimerit", "C_pauza"]
                 ).to_csv(STRAT_CSV, index=False)


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    print(update(backfill=int(sys.argv[1]) if len(sys.argv) > 1 else 0), "ture", f"{time.time()-t0:.0f}s")
