"""Explorare 22: schimbarea strategiilor din mers (cu 11 numere).
Pornește de la metodele din explorare21 (fără cele „învățate” pe prima jumătate) și încearcă meta-strategii:
urmează-l pe cel mai bun recent, rămâi pe o metodă până ratează de m ori apoi schimbă, alege metoda care a nimerit
ultima, cea cu pauza curentă cea mai scurtă/lungă, contrarian etc. Parametrii se aleg pe jumătatea 1, verificare pe jumătatea 2.
Rulare din rădăcina repo-ului: python cercetare/explorare22_schimbare_strategii.py"""
import numpy as np, pandas as pd

src = open("cercetare/explorare21_cautare_11.py").read().split("\ndef maxpause")[0]
exec(src)                                   # construiește H, A, B
names = [n for n in H if not n.startswith(("deplasări", "vot"))]
M = np.array([H[n] for n in names])        # (metode, ture) — nimerit da/nu
nm, nt = M.shape
rng = np.random.default_rng(1)


def maxpause(h):
    r = m = 0
    for v in h:
        r = 0 if v else r + 1; m = max(m, r)
    return m


def meta(kind, p):
    """Alege, la fiecare tură, o metodă folosind doar rezultatele anterioare; întoarce nimeririle."""
    out = np.zeros(nt, bool)
    cs = np.zeros(nm); hist = []              # nimeriri în ultimele p ture
    run = np.zeros(nm, int)                    # ture de la ultima nimerire, pe metodă
    cur, miss = 0, 0
    for t in range(nt):
        if kind == "cel mai bun recent":
            cur = int(np.argmax(cs + rng.random(nm) * 1e-6))
        elif kind == "cel mai slab recent":
            cur = int(np.argmin(cs + rng.random(nm) * 1e-6))
        elif kind == "a nimerit ultima (cea mai scurtă pauză)":
            cur = int(np.argmin(run + rng.random(nm) * 1e-6))
        elif kind == "cea mai lungă pauză (restanța)":
            cur = int(np.argmax(run + rng.random(nm) * 1e-6))
        elif kind == "rămâi până ratezi de p ori, apoi cel mai bun recent (100)":
            if miss >= p:
                cur = int(np.argmax(cs + rng.random(nm) * 1e-6)); miss = 0
        elif kind == "rămâi până ratezi de p ori, apoi la întâmplare":
            if miss >= p:
                cur = int(rng.integers(nm)); miss = 0
        elif kind == "rămâi până ratezi de p ori, apoi următoarea din listă":
            if miss >= p:
                cur = (cur + 1) % nm; miss = 0
        h = M[cur, t]; out[t] = h
        miss = 0 if h else miss + 1
        run = np.where(M[:, t], 0, run + 1)
        w = p if kind in ("cel mai bun recent", "cel mai slab recent") else 100
        cs += M[:, t]; hist.append(t)
        if len(hist) > w:
            cs -= M[:, hist.pop(0)]
    return out


rows = []
for kind, ps in [("cel mai bun recent", [1, 3, 5, 10, 17, 34, 100, 500, 2000]),
                 ("cel mai slab recent", [5, 17, 100, 500]),
                 ("a nimerit ultima (cea mai scurtă pauză)", [0]),
                 ("cea mai lungă pauză (restanța)", [0]),
                 ("rămâi până ratezi de p ori, apoi cel mai bun recent (100)", [1, 2, 3, 4]),
                 ("rămâi până ratezi de p ori, apoi la întâmplare", [1, 2, 3, 4]),
                 ("rămâi până ratezi de p ori, apoi următoarea din listă", [1, 2, 3, 4])]:
    for p in ps:
        h = meta(kind, p)
        rows.append((f"{kind}" + (f", p={p}" if p else ""), 100 * h[A].mean(), 100 * h[B].mean(),
                     maxpause(h[A]), maxpause(h[B]), int(sum(1 for i in range(1, len(h)) if h[i] and not h[i-1]) and 0)))
base = [("o singură metodă: cea mai bună pe jumătatea 1 (fixă)", None)]
bestA = max(names, key=lambda n: H[n][A].mean())
rows.append((f"fixă: {bestA}", 100 * H[bestA][A].mean(), 100 * H[bestA][B].mean(), maxpause(H[bestA][A]), maxpause(H[bestA][B]), 0))
rows.append(("fixă: 11 numere la întâmplare", 100 * H["11 numere la întâmplare"][A].mean(), 100 * H["11 numere la întâmplare"][B].mean(),
             maxpause(H["11 numere la întâmplare"][A]), maxpause(H["11 numere la întâmplare"][B]), 0))
R = pd.DataFrame(rows, columns=["meta-strategie", "jum1 %", "jum2 %", "pauza max jum1", "pauza max jum2", "_"]).drop(columns="_")
R = R.sort_values("jum1 %", ascending=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 70)
print(f"{nm} metode de bază, {len(R)} meta-strategii; {A.sum():,} + {B.sum():,} ture; hazard 55%\n")
print(R.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(f"\nCea mai bună la căutare: {R.iloc[0]['meta-strategie']} → {R.iloc[0]['jum1 %']:.2f}% / verificare {R.iloc[0]['jum2 %']:.2f}%")
print(f"Media pe jumătatea 2: {R['jum2 %'].mean():.2f}%, între {R['jum2 %'].min():.2f}% și {R['jum2 %'].max():.2f}%")
print(f"Corelație jum1 ↔ jum2: {np.corrcoef(R['jum1 %'], R['jum2 %'])[0,1]:+.2f}")
