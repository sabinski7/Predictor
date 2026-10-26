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


# ---------- strategia cu 11 numere ----------
S11 = ROOT / "strategie_11.csv"
K11 = 11


def numere_11(rule_preds, props, hist_numbers):
    """numerele celor 10 reguli (în ordinea din top, fără dubluri), apoi propunerile notebook-ului,
    apoi, dacă tot nu sunt 11, cele mai frecvente numere din ultimele 100 de extrageri."""
    nums = list(dict.fromkeys(list(rule_preds) + list(props)))[:K11]
    if len(nums) < K11:
        freq = np.bincount(np.asarray(hist_numbers[-100:], int), minlength=21)[1:]
        nums += [int(v) for v in np.argsort(-freq, kind="stable") + 1 if v not in nums][:K11 - len(nums)]
    return [int(v) for v in nums]


def update_s11(root=ROOT, path=S11):
    """Reface strategie_11.csv din top_reguli.csv și jurnal: cele 11 numere pentru fiecare tură,
    numărul ieșit, nimerit da/nu și pauza (ture la rând fără nimerire). Întoarce ultimul rând."""
    top_ = pd.read_csv(root / "top_reguli.csv", dtype=str)
    lg = pd.read_csv(root / "predictions_log.csv", dtype=str)
    lg = lg[lg.k == "9"].drop_duplicates("slot", keep="last").set_index("slot")
    h = pd.read_csv(root / "history.csv", dtype=str)
    hslots = (h.Date + " " + h.Time).tolist(); hnums = h["Drawn Number"].astype(int).to_numpy()
    pos = {s: i for i, s in enumerate(hslots)}
    rows, run = [], 0
    for slot, g in top_.groupby("slot", sort=True):
        g = g.sort_values("loc", key=lambda c: c.astype(int))
        props = [int(v) for v in lg.loc[slot, "propuneri"].split()] if slot in lg.index else []
        before = hnums[:pos[slot]] if slot in pos else hnums
        nums = numere_11(g.prezis.astype(int), props, before)
        res = g.iesit.iloc[0]
        if isinstance(res, str) and res:
            hit = int(res) in nums
            run = 0 if hit else run + 1
            rows.append([slot, " ".join(map(str, nums)), res, "da" if hit else "nu", run])
        else:
            rows.append([slot, " ".join(map(str, nums)), "", "", run])
    out = pd.DataFrame(rows, columns=["slot", "numere", "iesit", "nimerit", "pauza"])
    out.to_csv(path, index=False)
    return out


def message_11(root=ROOT):
    s = update_s11(root)
    done = s[s.iesit != ""]
    if not len(s) or s.iesit.iloc[-1] != "":
        return ""
    last = s.iloc[-1]
    pz = done.pauza.astype(int)
    rate = (done.nimerit == "da").mean()
    return (f"🎯 Strategia 11 numere → {last.slot}\n{last.numere.replace(' ', ', ')}\n"
            f"Pauza acum: {int(last.pauza)} ture fără nimerire · cea mai lungă: {pz.max()} · "
            f"nimerire {100*rate:.1f}% din {len(done)} ture (hazard 55%)")


ZILNIC = ROOT / "top_reguli_zilnic.csv"


def update_daily(root=ROOT, src=CSV, path=ZILNIC):
    """Rezumat pe zile al strategiei „top 10 reguli”: câte ture au fost jucate, în câte a nimerit
    măcar o regulă, câte nimeriri au fost în total (o tură poate avea mai multe) și procentul."""
    t = pd.read_csv(src, dtype=str)
    t = t[t.iesit.fillna("") != ""]
    if not len(t):
        return None
    t["data"] = t.slot.str[:10]
    per_tura = t.groupby(["data", "slot"]).nimerit.apply(lambda s: (s == "da").sum()).reset_index(name="n")
    z = per_tura.groupby("data").agg(ture=("slot", "count"),
                                     ture_cu_nimerire=("n", lambda v: int((v > 0).sum())),
                                     nimeriri_reguli=("n", "sum")).reset_index()
    z["procent_ture_cu_nimerire"] = (100 * z.ture_cu_nimerire / z.ture).round(1)
    z["data"] = pd.to_datetime(z.data).dt.strftime("%d.%m.%Y")
    z.to_csv(path, index=False)
    return z


if __name__ == "__main__":
    print(message())
    print(f"top_reguli.csv: {update_csv()} ture noi")
    print(update_daily().tail())
