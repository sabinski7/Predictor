"""Top 10 reguli din propuneri cu cea mai scurtă pauză maximă între nimeriri (pe jurnal),
recalculat după fiecare tură și trimis pe Telegram cu numărul prezis pentru tura următoare.

Reguli: orice combinație cu semne ± de 1–4 propuneri, sau ±2·pi, plus o constantă (mod 20) → 57.360 de reguli.
Pauza maximă include și turele până la prima nimerire și pauza în curs; la egalitate câștigă mai multe nimeriri.
Doar turele cu 9 propuneri și rezultat cunoscut. Același calcul ca în cercetare/explorare18_pauze_scurte.py.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOP = 10


def _coefs():
    rules = []
    for r in range(1, 5):
        for S in itertools.combinations(range(9), r):
            for sg in itertools.product([1, -1], repeat=r):
                co = np.zeros(9, int); co[list(S)] = sg
                rules.append(co)
    for i in range(9):
        for s in (2, -2):
            co = np.zeros(9, int); co[i] = s
            rules.append(co)
    return np.array(rules)


C = _coefs()


def name(f):
    co, c = C[f // 20], f % 20
    pos = [f"{'2·' if k == 2 else ''}p{i+1}" for i, k in enumerate(co) if k > 0]
    neg = [f"{'2·' if k == -2 else ''}p{i+1}" for i, k in enumerate(co) if k < 0]
    s = " + ".join(pos) + "".join(f" − {p}" for p in neg) if pos else "−" + " − ".join(neg)
    c = c if c <= 10 else c - 20
    return s if c == 0 else f"{s} {'+' if c > 0 else '−'} {abs(c)}"


def predict(f, props):
    return int((np.asarray(props) @ C[f // 20] + f % 20 - 1) % 20 + 1)


def load(root=ROOT):
    """(jurnal rezolvat, propunerile turei următoare sau None)."""
    h = pd.read_csv(root / "history.csv", dtype=str)
    res = dict(zip(h.Date + " " + h.Time, h["Drawn Number"].astype(int)))
    lg = pd.read_csv(root / "predictions_log.csv", dtype=str)
    lg = lg[lg.k == "9"].drop_duplicates("slot", keep="last")
    nxt = lg[~lg.slot.isin(res)].tail(1)
    lg = lg[lg.slot.isin(res)].sort_values("slot").copy()
    lg["rez"] = lg.slot.map(res)
    return lg, (nxt.slot.iloc[0], [int(v) for v in nxt.propuneri.iloc[0].split()]) if len(nxt) else None


def stats(lg):
    """pentru fiecare regulă: pauza maximă (cu cea în curs), nimeriri, pauza în curs."""
    P = np.array([[int(v) for v in p.split()] for p in lg.propuneri])
    y = lg.rez.to_numpy(int)
    H = (((y[:, None] - P @ C.T) % 20)[:, :, None] == np.arange(20)).reshape(len(y), -1)
    run = np.zeros(H.shape[1], int); mx = np.zeros(H.shape[1], int)
    for t in range(len(y)):
        mx = np.maximum(mx, np.where(H[t], run, 0))
        run = np.where(H[t], 0, run + 1)
    return np.maximum(mx, run), H.sum(0), run


def top(lg, k=TOP):
    mx, hits, run = stats(lg)
    order = np.lexsort((-hits, mx))[:k]
    return [(f, mx[f], hits[f], run[f]) for f in order]


def message(root=ROOT):
    lg, nxt = load(root)
    if nxt is None or len(lg) < 50:
        return ""
    now = top(lg)
    prev = top(lg.iloc[:-1])
    last = lg.iloc[-1]
    last_props = [int(v) for v in last.propuneri.split()]
    hit_prev = [i + 1 for i, (f, *_) in enumerate(prev) if predict(f, last_props) == last.rez]
    slot, props = nxt
    lines = [f"🏆 Top {TOP} reguli cu pauze scurte (jurnal: {len(lg)} ture) → {slot}"]
    for i, (f, mx, hits, run) in enumerate(now, 1):
        lines.append(f"{i}. {name(f)} → {predict(f, props)}  (pauza max {mx}, acum {run}, nimeriri {hits})")
    lines.append(f"La {last.slot[-5:]} a ieșit {last.rez}: "
                 + (f"✅ au nimerit regulile {', '.join(map(str, hit_prev))} din top-ul de atunci" if hit_prev else "❌ nicio regulă din top"))
    old, new = {p[0] for p in prev}, {p[0] for p in now}
    if old != new:
        lines.append("🔁 Intrat: " + ", ".join(name(f) for f in new - old) + "; ieșit: " + ", ".join(name(f) for f in old - new))
    cnt = pd.Series([predict(f, props) for f, *_ in now]).value_counts()
    rep = cnt[cnt > 1]
    if len(rep):
        lines.append("🔢 Numere repetate: " + ", ".join(f"{n} (×{c})" for n, c in rep.items()))
    lines.append("(Top-ul descrie trecutul; fiecare regulă are ~1 șansă din 20 la tura următoare.)")
    return "\n".join(lines)



CSV = ROOT / "top_reguli.csv"
START = 50          # primele 50 de ture din jurnal: top-ul nu are încă sens
COLS = ["slot", "loc", "regula", "prezis", "pauza_max", "pauza_acum", "nimeriri", "ture_jurnal", "iesit", "nimerit"]


def update_csv(root=ROOT, path=CSV):
    """Scrie în CSV top-ul de la fiecare tură, așa cum era ÎNAINTE de extragere (doar cu turele anterioare),
    cu numărul prezis de fiecare regulă; completează rezultatul când apare. Turele lipsă (inclusiv
    cele din trecut) sunt adăugate dintr-o singură trecere prin jurnal. Întoarce câte ture noi au fost scrise."""
    lg, nxt = load(root)
    old = pd.read_csv(path, dtype=str) if Path(path).exists() else pd.DataFrame(columns=COLS)
    done = set(old.slot)
    P = np.array([[int(v) for v in p.split()] for p in lg.propuneri])
    y = lg.rez.to_numpy(int)
    slots = lg.slot.tolist()
    H = (((y[:, None] - P @ C.T) % 20)[:, :, None] == np.arange(20)).reshape(len(y), -1) if len(y) else None
    run = np.zeros(len(C) * 20, int); mx = run.copy(); hits = run.copy()
    rows = []

    def emit(t, slot, props, res):
        eff = np.maximum(mx, run)
        for i, f in enumerate(np.lexsort((-hits, eff))[:TOP], 1):
            p = predict(f, props)
            rows.append([slot, i, name(f), p, eff[f], run[f], hits[f], t,
                         "" if res is None else res, "" if res is None else ("da" if p == res else "nu")])

    for t in range(len(y)):
        if t >= START and slots[t] not in done:
            emit(t, slots[t], P[t], int(y[t]))
        mx = np.maximum(mx, np.where(H[t], run, 0))
        run = np.where(H[t], 0, run + 1)
        hits += H[t]
    if nxt is not None and len(y) >= START and nxt[0] not in done:
        emit(len(y), nxt[0], nxt[1], None)

    # rezultate pentru turele care așteptau extragerea
    res = dict(zip(lg.slot, lg.rez.astype(int)))
    pend = (old.iesit.fillna("") == "") & old.slot.isin(res)
    if pend.any():
        old.loc[pend, "iesit"] = old.loc[pend, "slot"].map(res).astype(str)
        old.loc[pend, "nimerit"] = np.where(old.loc[pend, "prezis"].astype(int) == old.loc[pend, "slot"].map(res), "da", "nu")
    if not rows and not pend.any():
        return 0
    out = pd.concat([old, pd.DataFrame(rows, columns=COLS).astype(str)], ignore_index=True)
    out = out.sort_values(["slot", "loc"], key=lambda s: s.astype(int) if s.name == "loc" else s, kind="stable")
    out.to_csv(path, index=False)
    return len({r[0] for r in rows})


if __name__ == "__main__":
    print(message())
    print(f"top_reguli.csv: {update_csv()} ture noi")
