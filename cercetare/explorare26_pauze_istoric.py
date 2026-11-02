"""Explorare 26: reguli doar pe istoric (fără propuneri) cu cea mai mică pauză între nimeriri.
Intrări: numărul ieșit acum L ture, n1…n17 (o zi întreagă de extrageri), plus n34 și n51
(aceeași oră acum 2 și 3 zile). Reguli: combinații cu semne ± de 1–4 numere, ±2·nL, plus o constantă
(mod 20), de exemplu „n2 + n5 − n17 + 3” → aproximativ 1,4 milioane de reguli.
Pentru fiecare regulă: pauza maximă (ture la rând fără nimerire, inclusiv de la început și pauza în curs) și nimeririle,
pe tot istoricul, pe fiecare jumătate și pe ultimele 600 de ture (cât jurnalul tău).
Control: aceleași calcule cu numerele ieșite amestecate, ca să vedem ce pauze dă hazardul singur.
Rulare din rădăcina repo-ului: python cercetare/explorare26_pauze_istoric.py"""
import itertools
import numpy as np, pandas as pd

LAGS = list(range(1, 18)) + [34, 51]
h = pd.read_csv("history.csv")
x = h["Drawn Number"].to_numpy(); slots = (h.Date + " " + h.Time).to_numpy()
S0 = max(LAGS)
X = np.stack([x[S0 - L:len(x) - L] for L in LAGS], 1)          # X[t] = (n1, n2, …) pentru tura S0+t
Y = x[S0:]
T = len(Y)

coefs = []
for r in range(1, 5):
    for S in itertools.combinations(range(len(LAGS)), r):
        for sg in itertools.product([1, -1], repeat=r):
            co = np.zeros(len(LAGS), np.int16); co[list(S)] = sg; coefs.append(co)
for i in range(len(LAGS)):
    for s in (2, -2):
        co = np.zeros(len(LAGS), np.int16); co[i] = s; coefs.append(co)
C = np.array(coefs)
NB = len(C)


def name(b, c):
    co = C[b]; lab = lambda i, k: f"{'2·' if abs(k) == 2 else ''}n{LAGS[i]}"
    pos = [lab(i, k) for i, k in enumerate(co) if k > 0]; neg = [lab(i, k) for i, k in enumerate(co) if k < 0]
    s = " + ".join(pos) + "".join(f" − {p}" for p in neg) if pos else "−" + " − ".join(neg)
    c = c if c <= 10 else c - 20
    return s if c == 0 else f"{s} {'+' if c > 0 else '−'} {abs(c)}"


def gap_stats(D):
    """D: (ture, reguli-bază) = constanta c care ar fi nimerit la fiecare tură.
    Întoarce (pauza maximă, nimeriri), fiecare (baze, 20)."""
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
    red = np.maximum.reduceat(valT, starts)
    col = starts // n; cc = DsT[starts]
    mx[col, cc] = red
    sizes = np.diff(np.r_[starts, len(valT)]); cnt[col, cc] = sizes
    return mx, cnt


def scan(y, parts):
    """parts: dict nume -> slice de ture. Întoarce dict nume -> (mx, cnt) pentru toate regulile (NB, 20)."""
    out = {k: (np.zeros((NB, 20), np.int64), np.zeros((NB, 20), np.int64)) for k in parts}
    CH = 800
    for b0 in range(0, NB, CH):
        base = X @ C[b0:b0 + CH].T.astype(np.int64)
        D = ((y[:, None] - base) % 20).astype(np.int8)
        for k, sl in parts.items():
            mx, cnt = gap_stats(D[sl])
            out[k][0][b0:b0 + CH] = mx; out[k][1][b0:b0 + CH] = cnt
    return out


half = T // 2
parts = {"tot": slice(0, T), "jum1": slice(0, half), "jum2": slice(half, T), "ult600": slice(T - 600, T)}
R = scan(Y, parts)
print(f"Istoric: {T:,} ture (de la a {S0+1}-a); {NB*20:,} reguli pe n1…n17, n34, n51; șansa pură 5%\n")


def top(mx, cnt, k=10):
    key = mx.ravel() * 1_000_000 - cnt.ravel()
    return np.argsort(key, kind="stable")[:k]


nxt = np.array([x[len(x) - L] for L in LAGS])                  # intrările pentru tura următoare
def pred(f):
    b, c = divmod(f, 20)
    return int((nxt @ C[b] + c - 1) % 20 + 1)


for part, title in [("tot", "pe tot istoricul"), ("ult600", "pe ultimele 600 de ture (cât jurnalul tău)")]:
    mx, cnt = R[part]; n = parts[part].stop - parts[part].start
    print(f"TOP 10 cele mai scurte pauze {title}")
    print(f"   {'regula':<34}{'pauza max':>10}{'nimeriri':>15}{'tura următoare':>16}")
    for f in top(mx, cnt):
        b, c = divmod(f, 20)
        print(f"   {name(b, c):<34}{mx[b, c]:>10}{cnt[b, c]:>8} ({100*cnt[b,c]/n:4.1f}%){pred(f):>12}")
    print(f"   o regulă obișnuită are aici pauza maximă ~{int(np.median(mx))}\n")

mx1, c1 = R["jum1"]; mx2, c2 = R["jum2"]
best = top(mx1, c1, 1000)
print("Căutare pe jumătatea 1 → verificare pe jumătatea 2")
for f in best[:10]:
    b, c = divmod(f, 20)
    print(f"   {name(b, c):<34} jum1: pauza max {mx1[b,c]:>4}, {100*c1[b,c]/half:4.2f}%   →   jum2: pauza max {mx2[b,c]:>4}, {100*c2[b,c]/(T-half):4.2f}%")
print(f"   primele 1000 la căutare: pauza max medie {mx1.ravel()[best].mean():.0f} → pe jumătatea 2 {mx2.ravel()[best].mean():.0f} "
      f"(o regulă oarecare: {np.median(mx2):.0f}); nimerire {100*c1.ravel()[best].mean()/half:.2f}% → {100*c2.ravel()[best].mean()/(T-half):.2f}%")
print(f"   corelație pauza max jumătatea 1 ↔ 2 între reguli: {np.corrcoef(mx1.ravel(), mx2.ravel())[0,1]:+.3f}")

# control: numere amestecate
rng = np.random.default_rng(0)
Rz = scan(rng.permutation(Y), {"tot": slice(0, T), "ult600": slice(T - 600, T)})
for part in ("tot", "ult600"):
    print(f"\nControl ({part}): cea mai scurtă pauză maximă cu numere amestecate = {Rz[part][0].min()}, "
          f"real = {R[part][0].min()}; a 10-a: amestecat {np.sort(Rz[part][0].ravel())[9]}, real {np.sort(R[part][0].ravel())[9]}")
print(f"\nTura următoare = după {slots[-1]}")
