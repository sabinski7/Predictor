"""Simulator Win for Life Classico, doar pentru cercetare: produce extrageri cu aceeași structură și aceleași
proprietăți statistice ca jocul real (NU copiază generatorul operatorului și nu îi prezice numerele).

Structura, verificată pe combinatii.csv: la fiecare concurs se extrag 10 numere distincte din 1–20 (afișate
crescător) și, separat, Numerone din 1–20 (Numerone e printre cele 10 în ~50% din cazuri → extragere independentă).

Folosire:
    from simulator_wfl import SimulatorWFL
    sim = SimulatorWFL(seed=1)
    sim.extragere()          # ([10 numere], numerone)
    sim.serie_numerone(10000)
Rulare directă: validează simulatorul față de istoricul real (python cercetare/simulator_wfl.py, din rădăcina repo-ului).
"""
import numpy as np


class SimulatorWFL:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)          # PCG64; seed=None → sămânță din sistemul de operare

    def extragere(self):
        comb = np.sort(self.rng.choice(20, 10, replace=False) + 1)
        numerone = int(self.rng.integers(1, 21))
        return [int(v) for v in comb], numerone

    def serie(self, n):
        return [self.extragere() for _ in range(n)]

    def serie_numerone(self, n):
        return np.array([self.extragere()[1] for _ in range(n)])


if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.insert(0, "cercetare")
    src = open("cercetare/explorare23_comparatie_generatoare.py").read().split("\nnames = [")[0]
    exec(src)                                           # fingerprint(), real, N, K
    ref = pd.DataFrame([fingerprint(np.random.default_rng(1000 + i).integers(0, K, N)) for i in range(300)])
    mu, sd = ref.mean(), ref.std()
    dist = lambda x: float(np.sqrt((((pd.Series(fingerprint(x)) - mu) / sd) ** 2).mean()))
    sims = [dist(SimulatorWFL(s).serie_numerone(N) - 1) for s in range(20)]
    print("1) Numerone, 10.000 de extrageri, amprentă statistică (distanța față de ideal; ~1 = aleator perfect)")
    print(f"   extragerile reale: {dist(real):.2f}   simulatorul (20 de serii): mediană {np.median(sims):.2f}, "
          f"între {min(sims):.2f} și {max(sims):.2f}")

    c = pd.read_csv("combinatii.csv")
    comb = c[[f"n{i}" for i in range(1, 11)]].to_numpy(); nm = c.numerone.to_numpy()
    def comb_stats(comb, nm):
        return {
            "Numerone printre cele 10 %": 100 * np.mean([nm[i] in comb[i] for i in range(len(nm))]),
            "suma celor 10": comb.sum(1).mean(),
            "abaterea sumei": comb.sum(1).std(),
            "numere comune cu concursul anterior": np.mean([len(set(comb[i]) & set(comb[i - 1])) for i in range(1, len(comb))]),
            "perechi consecutive (ex. 4,5) în cele 10": np.mean((np.diff(comb, axis=1) == 1).sum(1)),
            "numere pare în cele 10": np.mean((comb % 2 == 0).sum(1)),
        }
    real_c = comb_stats(comb, nm)
    sim_c = []
    for s in range(1000):
        e = SimulatorWFL(s).serie(len(c))
        sim_c.append(comb_stats(np.array([a for a, _ in e]), np.array([b for _, b in e])))
    S = pd.DataFrame(sim_c)
    print(f"\n2) Combinațiile complete: {len(c)} concursuri reale față de 1.000 de serii simulate de aceeași lungime")
    print(f"   {'măsura':<44}{'real':>8}{'simulat (medie)':>17}{'90% din simulări':>20}")
    for k, v in real_c.items():
        lo, hi = S[k].quantile(.05), S[k].quantile(.95)
        print(f"   {k:<44}{v:8.2f}{S[k].mean():17.2f}{f'{lo:.2f} – {hi:.2f}':>20}  {'✔' if lo <= v <= hi else '✘'}")
    print("\n   Exemplu, 3 extrageri simulate:")
    sim = SimulatorWFL()
    for _ in range(3):
        a, b = sim.extragere(); print(f"   {' '.join(f'{v:2d}' for v in a)}   Numerone {b}")
