"""Explorare 23: seamănă extragerile cu un generator aleator obișnuit?
Comparăm ultimele 10.000 de extrageri cu secvențe de 10.000 de numere 1–20 produse de generatoare cunoscute
(bune: Mersenne Twister, PCG64, secrets/OS; slabe intenționat: LCG mic, „RANDU”, LCG cu modulo pe biții de jos,
xorshift trunchiat, un generator cu „memorie” care evită repetițiile). Comparația e statistică (amprente),
NU o încercare de a găsi starea sau sămânța generatorului operatorului.
Rulare din rădăcina repo-ului: python cercetare/explorare23_comparatie_generatoare.py"""
import random, secrets
import numpy as np, pandas as pd
from scipy import stats

N, K = 10000, 20
real = pd.read_csv("history.csv")["Drawn Number"].to_numpy()[-N:] - 1


def lcg(a, c, m, seed, shift=0):
    s = seed
    def nxt():
        nonlocal s
        s = (a * s + c) % m
        return s >> shift
    return nxt


def gen(name, seed):
    r = np.random.default_rng(seed)
    if name == "Mersenne Twister (Python random)":
        g = random.Random(seed); return np.array([g.randrange(K) for _ in range(N)])
    if name == "PCG64 (numpy)":
        return r.integers(0, K, N)
    if name == "secrets (sistemul de operare)":
        return np.array([secrets.randbelow(K) for _ in range(N)])
    if name == "LCG bun, modulo pe biții de sus":
        f = lcg(6364136223846793005, 1442695040888963407, 2**64, seed, 33); return np.array([f() % K for _ in range(N)])
    if name == "LCG slab: biții de jos (mod 2^16)":
        f = lcg(1103515245, 12345, 2**16, seed); return np.array([f() % K for _ in range(N)])
    if name == "RANDU (istoric, defect)":
        f = lcg(65539, 0, 2**31, seed | 1); return np.array([f() * K >> 31 for _ in range(N)])
    if name == "LCG cu perioadă mică (m=4096)":
        f = lcg(1029, 221, 4096, seed); return np.array([f() * K // 4096 for _ in range(N)])
    if name == "„evită repetițiile” (+20% schimbare)":
        out = [int(r.integers(K))]
        for _ in range(N - 1):
            v = int(r.integers(K))
            if v == out[-1] and r.random() < 0.2:
                v = int(r.integers(K))
            out.append(v)
        return np.array(out)
    if name == "„cald” ușor (numărul anterior +1% șansă)":
        out = [int(r.integers(K))]
        for _ in range(N - 1):
            out.append(out[-1] if r.random() < 0.01 else int(r.integers(K)))
        return np.array(out)


def fingerprint(x):
    """amprente statistice; fiecare = cât de departe e secvența de ideal (mai mic = mai aleator)."""
    f = {}
    f["frecvențe (chi²)"] = stats.chisquare(np.bincount(x, minlength=K)).statistic
    f["perechi consecutive (chi²)"] = stats.chisquare(np.bincount(x[:-1] * K + x[1:], minlength=K * K)).statistic
    f["perechi la 17 ture (chi²)"] = stats.chisquare(np.bincount(x[:-17] * K + x[17:], minlength=K * K)).statistic
    f["repetări imediate %"] = 100 * np.mean(x[1:] == x[:-1])
    gaps = np.concatenate([np.diff(np.flatnonzero(x == k)) for k in range(K)])
    f["pauza medie a unui număr"] = gaps.mean()
    f["pauza max a unui număr"] = gaps.max()
    up = np.sign(np.diff(x)); up = up[up != 0]
    f["serii sus/jos (nr.)"] = 1 + np.sum(up[1:] != up[:-1])
    f["autocorelație lag 1"] = np.corrcoef(x[:-1], x[1:])[0, 1]
    return f


names = ["Mersenne Twister (Python random)", "PCG64 (numpy)", "secrets (sistemul de operare)", "LCG bun, modulo pe biții de sus",
         "LCG slab: biții de jos (mod 2^16)", "RANDU (istoric, defect)", "LCG cu perioadă mică (m=4096)",
         "„evită repetițiile” (+20% schimbare)", "„cald” ușor (numărul anterior +1% șansă)"]
# distribuția „ideală” a fiecărei amprente: 300 de secvențe PCG64
ref = pd.DataFrame([fingerprint(np.random.default_rng(1000 + i).integers(0, K, N)) for i in range(300)])
mu, sd = ref.mean(), ref.std()


def distance(f):
    z = (pd.Series(f) - mu) / sd
    return z, float(np.sqrt((z ** 2).mean()))


rows = {}
zr, dr = distance(fingerprint(real))
rows["EXTRAGERILE REALE (ultimele 10.000)"] = dict(zr.round(2), distanta=round(dr, 2))
for nm in names:
    ds = [distance(fingerprint(gen(nm, 7 + s)))[1] for s in range(5)]
    z, _ = distance(fingerprint(gen(nm, 7)))
    rows[nm] = dict(z.round(2), distanta=round(float(np.mean(ds)), 2))
R = pd.DataFrame(rows).T
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("Abateri (z) față de un generator ideal; |z| > 3 = amprentă anormală. „distanța” = media pătratică (≈1 pentru aleator perfect).\n")
print(R.to_string())
ideal = np.sqrt((((ref - mu) / sd) ** 2).mean(axis=1))
print(f"\nDistanța pentru 300 de secvențe ideale: mediană {ideal.median():.2f}, 95% sub {ideal.quantile(.95):.2f}")
print(f"Extragerile reale: {dr:.2f} → percentila {100*(ideal < dr).mean():.0f} printre secvențele ideale")
