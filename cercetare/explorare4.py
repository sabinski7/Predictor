"""Faza 4: baterie suplimentară de teste (structură zilnică, periodicitate, ceas, teste clasice RNG).
Fiecare test compară datele reale cu 1.000 de simulări perfect aleatoare (test de permutare / Monte Carlo),
deci p-value-urile nu depind de aproximări."""
import lzma
import numpy as np, pandas as pd
from scipy.stats import chisquare, chi2_contingency

rng = np.random.default_rng(42)
df = pd.read_csv('history.csv')
df['dt'] = pd.to_datetime(df.Date + ' ' + df.Time)
df = df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
x = df['Drawn Number'].to_numpy() - 1
N = len(x); K = 20; SIMS = 1000
day_id = pd.factorize(df.Date)[0]
full_days = [np.flatnonzero(day_id == d) for d in range(day_id.max() + 1)]
full_days = [idx for idx in full_days if len(idx) == 17]
D = np.array([x[idx] for idx in full_days])          # (zile, 17)
print(f"N = {N:,} extrageri, {len(D):,} zile complete\n")

results = []
def mc(name, stat_fn, data_fn=None, two_sided=True):
    """p-value Monte Carlo: statistica reală vs SIMS serii iid uniforme de aceeași formă."""
    real = stat_fn(x if data_fn is None else data_fn(x))
    sims = []
    for _ in range(SIMS):
        xs = rng.integers(0, K, N)
        sims.append(stat_fn(xs if data_fn is None else data_fn(xs)))
    sims = np.array(sims)
    if two_sided:
        p = (np.sum(np.abs(sims - sims.mean()) >= abs(real - sims.mean())) + 1) / (SIMS + 1)
    else:
        p = (np.sum(sims >= real) + 1) / (SIMS + 1)
    results.append((name, real, sims.mean(), p))
    print(f"{name:<58} real {real:10.4f}  aleator {sims.mean():10.4f}  p = {p:.3f}")

days = lambda a: a[:len(D) * 17].reshape(len(D), 17) if a is not x else D

# --- A. structura zilnică
mc("A1 numere distincte pe zi (medie)", lambda d: np.mean([len(set(r)) for r in d]), days)
mc("A2 dispersia nr. distincte pe zi", lambda d: np.std([len(set(r)) for r in d]), days)
mc("A3 suprapunere zi / zi precedentă (numere comune)",
   lambda d: np.mean([len(set(a) & set(b)) for a, b in zip(d[:-1], d[1:])]), days)
mc("A4 repetări în aceeași zi la aceeași poziție ca ieri", lambda d: np.mean(d[1:] == d[:-1]), days)
mc("A5 max apariții ale unui număr într-o zi (medie)",
   lambda d: np.mean([np.bincount(r, minlength=K).max() for r in d]), days)
def pair_chi(d):
    co = np.zeros((K, K))
    for r in d:
        u = np.unique(r); co[np.ix_(u, u)] += 1
    iu = np.triu_indices(K, 1); v = co[iu]
    return ((v - v.mean()) ** 2 / v.mean()).sum()
mc("A6 perechi care apar împreună în aceeași zi (chi²)", pair_chi, days, two_sided=False)

# --- B. periodicitate
def spec_max(a):
    s = 0
    for k in range(K):
        b = (a == k).astype(float) - 1 / K
        s = max(s, (np.abs(np.fft.rfft(b)[1:]) ** 2).max() / len(a))
    return s
mc("B1 vârful spectral maxim (FFT, toate numerele)", spec_max, two_sided=False)
def spec_sum_at(a, periods=(17, 34, 119, 7 * 17)):
    s = 0
    for k in range(K):
        b = (a == k).astype(float) - 1 / K
        F = np.abs(np.fft.rfft(b)) ** 2 / len(a)
        for P in periods:
            s += F[int(round(len(a) / P))]
    return s
mc("B2 putere spectrală la perioade zi/săptămână", spec_sum_at, two_sided=False)

# --- C. legătura cu ceasul (RNG inițializat din timp?)
ts_hours = (df.dt.astype('int64') // 3_600_000_000_000).to_numpy() if df.dt.dtype == 'datetime64[ns]' else \
           (df.dt - pd.Timestamp('1970-01-01')) // pd.Timedelta(hours=1)
ts_hours = np.asarray(ts_hours)
doy = df.dt.dt.dayofyear.to_numpy(); dom = df.dt.dt.day.to_numpy(); hr = df.dt.dt.hour.to_numpy()
def clock_stat(a):
    best = 0
    for m in range(2, 41):
        t = np.zeros((m, K)); np.add.at(t, (ts_hours % m, a), 1)
        best = max(best, chi2_contingency(t)[0] / ((m - 1) * (K - 1)))
    for f in [(doy + hr) % K, (dom * hr) % K, (doy * 24 + hr) % K, (dom + hr) % K]:
        best = max(best, chisquare(np.bincount((a - f) % K, minlength=K))[0] / (K - 1))
    return best
mc("C1 relație cu timestamp-ul (39 moduli + 4 formule dată/oră)", clock_stat, two_sided=False)

# --- D. teste clasice RNG
mc("D1 autocorelație numerică max |r|, lag 1..200",
   lambda a: max(abs(np.corrcoef(a[:-l], a[l:])[0, 1]) for l in range(1, 201)), two_sided=False)
def poker(a):
    h = [len(set(a[i:i + 5])) for i in range(0, len(a) - 5, 5)]
    return np.bincount(h, minlength=6)[5] / len(h)
mc("D2 poker: blocuri de 5 cu toate diferite (fracție)", poker)
def coupon(a):
    lens, seen, c = [], set(), 0
    for v in a:
        c += 1; seen.add(v)
        if len(seen) == K: lens.append(c); seen, c = set(), 0
    return np.mean(lens)
mc("D3 colecționarul de cupoane: extrageri până apar toate 20", coupon)
mc("D4 maxim din blocuri de 17 (medie)", lambda a: a[:len(a) // 17 * 17].reshape(-1, 17).max(1).mean())
mc("D5 suma blocurilor de 17 (dispersie)", lambda a: a[:len(a) // 17 * 17].reshape(-1, 17).sum(1).std())
mc("D6 lungimea celei mai lungi serii crescătoare", lambda a: np.max(np.diff(np.flatnonzero(np.r_[True, np.diff(a) <= 0, True]))))
def triples(a):
    t = np.bincount(a[:-2] * 400 + a[1:-1] * 20 + a[2:], minlength=8000)
    return ((t - t.mean()) ** 2 / t.mean()).sum()
mc("D7 triplete consecutive (8.000 combinații, chi²)", triples, two_sided=False)
mc("D8 compresibilitate lzma (octeți)", lambda a: len(lzma.compress(a.astype(np.uint8).tobytes(), preset=9)))

ps = np.array([r[3] for r in results])
print(f"\n{len(ps)} teste. Cel mai mic p = {ps.min():.3f}; corectat (Bonferroni) = {min(1, ps.min() * len(ps)):.3f}")
print(f"Teste cu p < 0,01: {(ps < 0.01).sum()} (la întâmplare ne-am aștepta la ~{0.01 * len(ps):.1f})")
