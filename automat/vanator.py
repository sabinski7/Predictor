"""Vânătorul de reguli.

Caută automat reguli care ar prezice Numerone mai bine decât șansa și le acceptă doar dacă trec
trei etape, fiecare pe date pe care etapa anterioară nu le-a văzut:

  1. CĂUTARE   – ~22.000 de reguli, pe tot istoricul în afară de ultimele VALIDARE ture;
                 se rețin cele mai bune TOP_CAUTARE (după scorul z față de șansă);
  2. VALIDARE  – acele reguli, pe ultimele VALIDARE ture; trec doar cele cu p < ALFA / TOP_CAUTARE;
  3. PROBĂ     – regula trecută e înghețată și urmărită pe extrageri reale noi (care nu existau când
                 a fost aleasă). Confirmată când p < ALFA_PROBA (după cel puțin PROBA_MIN ture în care se
                 aplică); respinsă dacă, după PROBA_MIN, nu e peste șansă sau dacă trec PROBA_MAX ture.

Reguli pentru UN număr (șansa pură 5%) și seturi de 9 numere (șansa pură 45%).
Starea (candidații în probă, confirmați, respinși) e în automat/vanator_stare.json,
iar un raport lizibil în automat/vanator_raport.md.

Rulare manuală:  python automat/vanator.py   (rulează căutarea acum și afișează mesajele)
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "history.csv"
STATE = Path(__file__).resolve().parent / "vanator_stare.json"
REPORT = Path(__file__).resolve().parent / "vanator_raport.md"

K = 20
START = 1800            # primele ture servesc doar ca istoric pentru reguli
VALIDARE = 10000        # ultimele ture, păstrate pentru validare (destule și pentru regulile „la ora X”)
TOP_CAUTARE = 50
ALFA = 0.01
PROBA_MIN = 100         # ture în care regula se aplică, înainte de prima decizie
PROBA_MAX = 3400        # ~200 de zile: dacă nu e confirmată până atunci, e respinsă
ALFA_PROBA = 0.001      # prag de confirmare (strict, pentru că verificăm zilnic)
LAG_MAX = 60
MULT = [1, -1, 3, -3, 7, -7, 9, -9]          # unitățile modulo 20 (11 ≡ −9, 13 ≡ −7, 17 ≡ −3)
PAIR_MAX = 12
WINDOWS = [17, 34, 51, 119, 255, 510, 1020, 1700]
ORE = list(range(7, 24))


# ───────────────────────── date ─────────────────────────
def load():
    df = pd.read_csv(HISTORY, dtype={"Date": str, "Time": str})
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M")
    df = df.sort_values("dt").reset_index(drop=True)
    x = df["Drawn Number"].astype(int).to_numpy() - 1
    return df, x, df["dt"].dt.hour.to_numpy()


def next_hour(dt):
    n = dt + pd.Timedelta(hours=1)
    return 7 if n.hour > 23 or n.hour < 7 else n.hour


class Feats:
    """Mărimi calculate o singură dată pentru t = 0..N (t = N este extragerea următoare, necunoscută)."""
    def __init__(self, x, hours, nxt_hour):
        N = len(x)
        self.N, self.x = N, x
        self.h = np.r_[hours, nxt_hour]
        oh = np.eye(K, dtype=np.int32)[x]
        self.cs = np.vstack([np.zeros((1, K), np.int32), np.cumsum(oh, 0)])     # cs[t] = numărări în x[:t]
        last, gaps = np.full(K, -1), np.zeros((N + 1, K), np.int32)
        hc, hcount = np.zeros((24, K), np.int32), np.zeros((N + 1, K), np.int32)
        for t in range(N + 1):
            gaps[t] = t - last
            hcount[t] = hc[self.h[t]]
            if t < N:
                last[x[t]] = t
                hc[hours[t], x[t]] += 1
        self.gaps, self.hcount = gaps, hcount
        self._rank = {}

    def lag(self, L):
        """x[t-L] pentru t = 0..N (−1 unde nu există)."""
        out = np.full(self.N + 1, -1)
        out[L:] = self.x[: self.N + 1 - L]
        return out

    def window_order(self, W):
        """Pentru fiecare t: numerele ordonate de la cel mai frecvent la cel mai rar în ultimele W ture."""
        if W not in self._rank:
            t = np.arange(self.N + 1)
            c = self.cs[t] - self.cs[np.maximum(t - W, 0)]
            self._rank[W] = np.argsort(-c, axis=1, kind="stable")
        return self._rank[W]


# ───────────────────────── reguli ─────────────────────────
def describe(r):
    f = r["f"]
    c = r.get("c", 0)
    cs = f" {'+' if c <= 10 else '−'} {c if c <= 10 else K - c}" if c else ""
    if f == "lag":
        a = "" if r["a"] == 1 else ("−" if r["a"] == -1 else f"{r['a']} × ")
        return f"numărul = {a}(cel de acum {r['L']} ture){cs}"
    if f == "pair":
        s1 = "" if r["s1"] == 1 else "−"
        s2 = "+" if r["s2"] == 1 else "−"
        return f"numărul = {s1}(acum {r['A']}) {s2} (acum {r['B']}){cs}"
    if f == "zi":
        return f"numărul = cel de la aceeași oră acum {r['d']} zile{cs}"
    if f == "ora":
        return f"la ora {r['h']:02d}:00 joacă {r['n'] + 1}"
    if f == "rang":
        return f"numărul cu locul {r['r'] + 1} în frecvența ultimelor {r['W']} ture"
    if f == "gap":
        return f"numărul cu locul {r['r'] + 1} în întârziere"
    if f == "set_fereastra":
        return f"setul celor 9 {'cele mai calde' if r['cald'] else 'cele mai reci'} numere din ultimele {r['W']} ture"
    if f == "set_gap":
        return f"setul celor 9 numere {'cele mai întârziate' if r['mare'] else 'ieșite cel mai recent'}"
    if f == "set_ora":
        return f"setul celor 9 numere {'cele mai dese' if r['des'] else 'cele mai rare'} la ora extragerii"
    raise ValueError(f)


def rule_id(r):
    return json.dumps(r, sort_keys=True)


def predict(r, F):
    """Întoarce (tip, valori) pentru t = 0..N.
    tip 'num': valori = număr 0..19 sau −1 (regula nu se aplică); tip 'set': matrice bool (N+1, 20)."""
    f, N = r["f"], F.N
    if f == "lag":
        b = F.lag(r["L"])
        return "num", np.where(b < 0, -1, (r["a"] * b + r["c"]) % K)
    if f == "pair":
        a, b = F.lag(r["A"]), F.lag(r["B"])
        return "num", np.where((a < 0) | (b < 0), -1, (r["s1"] * a + r["s2"] * b + r["c"]) % K)
    if f == "zi":
        b = F.lag(17 * r["d"])
        return "num", np.where(b < 0, -1, (b + r["c"]) % K)
    if f == "ora":
        return "num", np.where(F.h == r["h"], r["n"], -1)
    if f == "rang":
        return "num", F.window_order(r["W"])[:, r["r"]]
    if f == "gap":
        return "num", np.argsort(-F.gaps, axis=1, kind="stable")[:, r["r"]]
    if f == "set_fereastra":
        o = F.window_order(r["W"])
        top = o[:, :9] if r["cald"] else o[:, -9:]
    elif f == "set_gap":
        o = np.argsort(-F.gaps if r["mare"] else F.gaps, axis=1, kind="stable")
        top = o[:, :9]
    elif f == "set_ora":
        o = np.argsort(-F.hcount if r["des"] else F.hcount, axis=1, kind="stable")
        top = o[:, :9]
    else:
        raise ValueError(f)
    m = np.zeros((N + 1, K), bool)
    np.put_along_axis(m, top, True, 1)
    return "set", m


def hits_of(r, F, ts):
    """Vector bool de nimeriri și vector bool „regula se aplică” pentru turele ts (toate < N)."""
    kind, v = predict(r, F)
    y = F.x[ts]
    if kind == "num":
        p = v[ts]
        return p == y, p >= 0
    return v[ts, y], np.ones(len(ts), bool)


def base_of(r):
    return 0.45 if r["f"].startswith("set") else 0.05


def all_rules():
    rs = []
    for L in range(1, LAG_MAX + 1):
        for a in MULT:
            rs += [{"f": "lag", "L": L, "a": a, "c": c} for c in range(K)]
    for A, B in itertools.combinations(range(1, PAIR_MAX + 1), 2):
        for s1, s2 in [(1, 1), (1, -1), (-1, 1)]:
            rs += [{"f": "pair", "A": A, "B": B, "s1": s1, "s2": s2, "c": c} for c in range(K)]
    for d in range(1, 15):
        rs += [{"f": "zi", "d": d, "c": c} for c in range(K)]
    rs += [{"f": "ora", "h": h, "n": n} for h in ORE for n in range(K)]
    rs += [{"f": "rang", "W": W, "r": r} for W in WINDOWS for r in range(K)]
    rs += [{"f": "gap", "r": r} for r in range(K)]
    rs += [{"f": "set_fereastra", "W": W, "cald": c} for W in WINDOWS for c in (True, False)]
    rs += [{"f": "set_gap", "mare": m} for m in (True, False)]
    rs += [{"f": "set_ora", "des": d} for d in (True, False)]
    return rs


def zscore(h, n, p):
    return (h - n * p) / np.sqrt(np.maximum(n * p * (1 - p), 1e-9))


def search(F, ts):
    """Scor z pe turele ts pentru toate regulile (vectorizat pe familii)."""
    y = F.x[ts]
    rules = all_rules()
    z = np.empty(len(rules))
    idx = 0
    # familiile cu constanta c (20 de reguli deodată)
    def fam_c(base_vals, valid):
        d = (y - base_vals) % K
        d = d[valid]
        cnt = np.bincount(d, minlength=K)
        n = valid.sum()
        return zscore(cnt, n, 0.05)
    for L in range(1, LAG_MAX + 1):
        b = F.lag(L)[ts]
        for a in MULT:
            z[idx: idx + K] = fam_c((a * b) % K, b >= 0); idx += K
    for A, B in itertools.combinations(range(1, PAIR_MAX + 1), 2):
        a, b = F.lag(A)[ts], F.lag(B)[ts]
        for s1, s2 in [(1, 1), (1, -1), (-1, 1)]:
            z[idx: idx + K] = fam_c((s1 * a + s2 * b) % K, (a >= 0) & (b >= 0)); idx += K
    for d in range(1, 15):
        b = F.lag(17 * d)[ts]
        z[idx: idx + K] = fam_c(b % K, b >= 0); idx += K
    hh = F.h[ts]
    for h in ORE:
        m = hh == h
        z[idx: idx + K] = zscore(np.bincount(y[m], minlength=K), m.sum(), 0.05); idx += K
    while idx < len(rules):                       # restul: direct, prin predict
        r = rules[idx]
        hit, ok = hits_of(r, F, ts)
        z[idx] = zscore(hit[ok].sum(), ok.sum(), base_of(r)); idx += 1
    return rules, z


# ───────────────────────── rulare zilnică ─────────────────────────
def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"ultima_zi": "", "candidati": [], "rulari": 0}


def save_state(s):
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=1))


def day_complete(df):
    """Ultima zi complet extrasă (dacă ultima extragere e 23:00, ziua curentă; altfel ziua precedentă)."""
    last = df["dt"].iloc[-1]
    d = last.normalize() if last.hour == 23 else last.normalize() - pd.Timedelta(days=1)
    return f"{d:%Y-%m-%d}"


def evaluate_live(c, F, df):
    """Rezultatele regulii pe extragerile de după înghețare."""
    after = np.flatnonzero(df["dt"].to_numpy() > np.datetime64(pd.Timestamp(c["inghetat_dupa"])))
    if not len(after):
        return 0, 0
    hit, ok = hits_of(c["regula"], F, after)
    return int(ok.sum()), int(hit[ok].sum())


def current_prediction(c, F):
    kind, v = predict(c["regula"], F)
    if kind == "num":
        return None if v[F.N] < 0 else str(v[F.N] + 1)
    return " ".join(str(i + 1) for i in np.flatnonzero(v[F.N]))


def run(force=False):
    """Rulează vânătorul dacă e o zi nouă completă (sau force). Întoarce lista de mesaje Telegram."""
    df, x, hours = load()
    state = load_state()
    zi = day_complete(df)
    msgs = []
    F = Feats(x, hours, next_hour(df["dt"].iloc[-1]))
    N = F.N

    # 1) actualizează candidații în probă / confirmați
    for c in state["candidati"]:
        if c["stare"] not in ("proba", "confirmat"):
            continue
        n, h = evaluate_live(c, F, df)
        c["proba_n"], c["proba_hits"] = n, h
        if c["stare"] != "proba" or n < PROBA_MIN:
            continue
        p = binomtest(h, n, c["sansa"], alternative="greater").pvalue
        c["proba_p"] = p
        if p < ALFA_PROBA:
            c["stare"] = "confirmat"
            msgs.append(f"🚨 REGULĂ CONFIRMATĂ\n{c['descriere']}\n"
                        f"Probă pe {n} extrageri noi: {100*h/n:.1f}% (șansa pură {100*c['sansa']:.0f}%), p = {p:.5f}.\n"
                        f"De acum, mesajele includ și ce prezice ea.")
        elif h / n <= c["sansa"] or n >= PROBA_MAX:
            c["stare"] = "respins"
            msgs.append(f"❌ Candidat respins după probă\n{c['descriere']}\n"
                        f"Pe {n} extrageri noi: {100*h/n:.1f}% (șansa pură {100*c['sansa']:.0f}%) — a fost noroc.")

    if not force and state["ultima_zi"] >= zi:
        save_state(state)
        return msgs

    # 2) căutare + validare pe datele de până acum
    ts_all = np.arange(START, N)
    ts_s, ts_v = ts_all[:-VALIDARE], ts_all[-VALIDARE:]
    rules, z = search(F, ts_s)
    order = np.argsort(-z)[:TOP_CAUTARE]
    known = {rule_id(c["regula"]) for c in state["candidati"]}
    rows, new_c = [], []
    for i in order:
        r = rules[i]
        hs, oks = hits_of(r, F, ts_s)
        hv, okv = hits_of(r, F, ts_v)
        ns, nv = int(oks.sum()), int(okv.sum())
        rs, rv = hs[oks].mean(), (hv[okv].mean() if nv else 0)
        pv = binomtest(int(hv[okv].sum()), nv, base_of(r), alternative="greater").pvalue if nv else 1
        rows.append((r, rs, rv, pv, ns, nv))
        if pv < ALFA / TOP_CAUTARE and rule_id(r) not in known:
            c = {"regula": r, "descriere": describe(r), "sansa": base_of(r), "stare": "proba",
                 "inghetat_dupa": f"{df['dt'].iloc[-1]:%Y-%m-%d %H:%M}", "cautare": round(rs, 4),
                 "validare": round(rv, 4), "validare_p": pv, "proba_n": 0, "proba_hits": 0}
            state["candidati"].append(c)
            known.add(rule_id(r))
            new_c.append((c, nv, rv, pv))

    if new_c:
        lines = [f"🔬 {len(new_c)} candidat(i) nou(i) în probă – încă NECONFIRMAȚI"]
        for c, nv, rv, pv in new_c[:5]:
            lines.append(f"• {c['descriere']}: validare {100*rv:.1f}% pe {nv} ture "
                         f"(șansa pură {100*c['sansa']:.0f}%), p = {pv:.5f}")
        if len(new_c) > 5:
            lines.append(f"• … și încă {len(new_c) - 5} (vezi automat/vanator_raport.md)")
        lines.append("Îi urmăresc pe extragerile următoare și te anunț când se confirmă sau se infirmă.")
        msgs.append("\n".join(lines))

    state["ultima_zi"] = zi
    state["rulari"] = state.get("rulari", 0) + 1
    state["ultima_cautare"] = {"reguli": len(rules), "ture_cautare": len(ts_s), "ture_validare": len(ts_v),
                               "cel_mai_bun_validare": max((rv for _, _, rv, _, _, _ in rows), default=0)}
    save_state(state)
    write_report(state, rows, len(rules), len(ts_s), len(ts_v), df)

    # 3) raport săptămânal (duminica)
    if pd.Timestamp(zi).dayofweek == 6:
        act = [c for c in state["candidati"] if c["stare"] == "proba"]
        conf = [c for c in state["candidati"] if c["stare"] == "confirmat"]
        resp = [c for c in state["candidati"] if c["stare"] == "respins"]
        msgs.append(f"🧭 Vânătorul de reguli – raport săptămânal\n"
                    f"{len(rules):,} reguli testate zilnic. În probă: {len(act)}, confirmate: {len(conf)}, "
                    f"respinse până acum: {len(resp)}.")
    return msgs


def confirmed_predictions():
    """Pentru mesajul de la fiecare extragere: ce prezic regulile confirmate (dacă există)."""
    state = load_state()
    conf = [c for c in state["candidati"] if c["stare"] == "confirmat"]
    if not conf:
        return ""
    df, x, hours = load()
    F = Feats(x, hours, next_hour(df["dt"].iloc[-1]))
    lines = []
    for c in conf:
        p = current_prediction(c, F)
        if p:
            lines.append(f"🚨 {c['descriere']} → {p}")
    return "\n".join(lines)


def write_report(state, rows, n_rules, n_s, n_v, df):
    L = [f"# Vânătorul de reguli – ultima rulare\n",
         f"Istoric până la **{df['dt'].iloc[-1]:%Y-%m-%d %H:%M}**. {n_rules:,} reguli; "
         f"căutare pe {n_s:,} ture, validare pe ultimele {n_v:,} ture (nevăzute la căutare).\n",
         f"Prag de validare: p < {ALFA / TOP_CAUTARE:.4f} (corectat pentru {TOP_CAUTARE} reguli). "
         f"Probă: pe extrageri noi, confirmare la p < {ALFA_PROBA} (minimum {PROBA_MIN} ture în care se aplică, maximum {PROBA_MAX}).\n",
         "## Cele mai bune reguli la căutare și cum s-au descurcat la validare\n",
         "| Regula | Șansa pură | Căutare | Validare | p validare |", "|---|---|---|---|---|"]
    for r, rs, rv, pv, _, _ in rows[:20]:
        L.append(f"| {describe(r)} | {100*base_of(r):.0f}% | {100*rs:.2f}% | {100*rv:.2f}% | {pv:.3f} |")
    L.append("\n## Candidați\n")
    if not state["candidati"]:
        L.append("Niciun candidat până acum.")
    else:
        L += ["| Regula | Stare | Înghețată după | Probă |", "|---|---|---|---|"]
        for c in state["candidati"]:
            pr = f"{c['proba_hits']}/{c['proba_n']}" + (f" ({100*c['proba_hits']/c['proba_n']:.1f}%)" if c["proba_n"] else "")
            L.append(f"| {c['descriere']} | {c['stare']} | {c['inghetat_dupa']} | {pr} |")
    REPORT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    import time
    t0 = time.time()
    for m in run(force=True):
        print(m, "\n")
    print(f"({time.time() - t0:.0f}s) raport: {REPORT}")
