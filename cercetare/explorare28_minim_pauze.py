"""Explorare 28: căutare țintită a strategiei cu cele mai scurte pauze între nimeriri.
Pornește de la cele 62 de metode de bază din explorare21 (modele, calde/reci, recență, aceeași oră, excluderi, Markov,
reguli), fiecare dând un clasament al celor 20 de numere la fiecare tură, doar din extragerile anterioare.
Încercări noi, toate alese pe jumătatea 1 a istoricului și verificate pe jumătatea 2:
  1) are jocul „memorie” a pauzelor? P(nimerire | câte ture la rând a ratat strategia înainte), pe toate metodele;
  2) politica „după stare”: la fiecare tură alege metoda care, în starea ei de acum (ture de la ultima nimerire),
     a nimerit cel mai des pe jumătatea 1;
  3) căutare exhaustivă de perechi „joacă metoda X; după s ratări la rând treci pe Y până nimerești”, optimizate direct
     pentru cea mai scurtă pauză maximă și cele mai puține pauze lungi;
  4) aceleași lucruri pentru un singur număr (5%).
Rezultatele se compară cu ce dă hazardul pentru aceeași șansă pe tură (simulări).
Rulare din rădăcina repo-ului: python cercetare/explorare28_minim_pauze.py"""
import numpy as np, pandas as pd

src = open("cercetare/explorare21_cautare_11.py").read().split("\n# rangul numărului ieșit")[0]
exec(src)                                         # S (scoruri), idx, A, B, x, T0, N, NN, tie
names = [n for n in S if not n.startswith(("deplasări",))]
y = x[idx]
RANK = np.empty((len(names), len(idx)), np.int8)
for i, nm in enumerate(names):
    sc = S[nm][idx] + tie
    RANK[i] = (sc > sc[np.arange(len(idx)), y][:, None]).sum(1)
rng = np.random.default_rng(0)
RANK = np.vstack([RANK, rng.integers(0, 20, len(idx))[None, :]])       # + o metodă pur aleatoare (control)
names.append("la întâmplare")
nm_, nt = RANK.shape
iA, iB = np.flatnonzero(A), np.flatnonzero(B)


def pauses(h):
    out, r = [], 0
    for v in h:
        if v:
            out.append(r); r = 0
        else:
            r += 1
    out.append(r)
    return np.array(out)


def null_maxpause(p, n, sims=2000):
    m = []
    for _ in range(sims):
        m.append(pauses(rng.random(n) < p).max())
    return np.array(m)


def describe(h, p, label):
    ps = pauses(h); nul = null_maxpause(p, len(h), 500)
    print(f"   {label:<58} nimerire {100*h.mean():5.1f}%  pauza max {ps.max():3d} "
          f"(hazard: de obicei {np.median(nul):.0f}, 90% ≤ {np.percentile(nul,90):.0f})  "
          f"pauze ≥{5 if p > .3 else 40}: {(ps >= (5 if p > .3 else 40)).sum()}")


for K, lab in [(11, "11 NUMERE (55% pe tură)"), (1, "UN SINGUR NUMĂR (5% pe tură)")]:
    p = K / 20
    H = RANK < K                                       # (metode, ture)
    print(f"\n================ {lab} ================")

    # 1) memorie a pauzelor
    run = np.zeros(nm_, int); TC = 100; tab = np.zeros((TC + 1, 2))
    for t in range(nt):
        st = np.minimum(run, TC)
        np.add.at(tab, (st, 0), H[:, t]); np.add.at(tab, (st, 1), 1)
        run = np.where(H[:, t], 0, run + 1)
    print("1) Șansa de nimerire după r ratări la rând (toate metodele, tot istoricul):")
    rs = range(0, 9) if K == 11 else range(0, 80, 10)
    print("   " + "  ".join(f"r={r}: {100*tab[r,0]/max(tab[r,1],1):.1f}%" for r in rs))

    # 2) politica după stare (starea proprie a fiecărei metode), învățată pe A
    run = np.zeros(nm_, int); CAP = 8 if K == 11 else 60
    hitsA = np.zeros((nm_, CAP + 1)); cntA = np.zeros((nm_, CAP + 1))
    chosen = np.zeros(nt, bool)
    for t in range(nt):
        st = np.minimum(run, CAP)
        if B[t]:
            rate = (hitsA[np.arange(nm_), st] + 1) / (cntA[np.arange(nm_), st] + 1 / p)
            chosen[t] = H[int(np.argmax(rate)), t]
        else:
            hitsA[np.arange(nm_), st] += H[:, t]; cntA[np.arange(nm_), st] += 1
        run = np.where(H[:, t], 0, run + 1)
    print("2) Politica „după stare” (învățată pe jumătatea 1, jucată pe jumătatea 2):")
    describe(chosen[iB], p, "politica după stare")

    # 3) perechi X → (după s ratări) Y, optimizate pe A pentru pauze
    def simulate(m1, m2, s, sl):
        h1, h2 = H[m1, sl], H[m2, sl]; out = np.empty(len(h1), bool); miss = 0
        for t in range(len(h1)):
            v = h2[t] if miss >= s else h1[t]
            out[t] = v; miss = 0 if v else miss + 1
        return out
    rateA = H[:, iA].mean(1)
    cand1 = np.argsort(-rateA)[:8]
    best = []
    for m1 in cand1:
        for m2 in range(nm_):
            for s in ((1, 2, 3, 4) if K == 11 else (10, 20, 40)):
                ps = pauses(simulate(m1, m2, s, iA))
                best.append((ps.max(), (ps >= (5 if K == 11 else 40)).sum(), m1, m2, s))
    best.sort()
    print(f"3) Cele mai bune perechi „X, apoi Y după s ratări” din {len(best):,} (alese pe jumătatea 1):")
    for mx, n5, m1, m2, s in best[:3]:
        hB = simulate(m1, m2, s, iB)
        print(f"   {names[m1][:28]} → {names[m2][:28]}, s={s}: jum1 pauza max {mx}")
        describe(hB, p, "   aceeași pereche pe jumătatea 2")

    # referințe
    print("Referințe pe jumătatea 2:")
    describe(H[-1, iB], p, "numere la întâmplare")
    bestA = int(np.argmax(rateA[:-1]))
    describe(H[bestA, iB], p, f"cea mai bună metodă pe jumătatea 1 ({names[bestA][:30]})")
