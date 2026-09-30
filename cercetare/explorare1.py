"""Faza 1: teste de aleatorism extinse."""
import numpy as np, pandas as pd
from scipy.stats import chisquare, chi2_contingency, chi2, norm

df = pd.read_csv('history.csv')
df['dt'] = pd.to_datetime(df.Date + ' ' + df.Time)
df = df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
x = df['Drawn Number'].to_numpy() - 1
N = len(x); K = 20
print("N =", N)

def ct(a, b, na, nb):
    t = np.zeros((na, nb)); np.add.at(t, (a, b), 1); return t

res = []
# serial dependence at lags 1..60
for lag in range(1, 61):
    t = ct(x[:-lag], x[lag:], K, K)
    p = chi2_contingency(t)[1]
    same = (x[:-lag] == x[lag:]).mean()
    res.append(('lag', lag, p, same))
lagp = np.array([r[2] for r in res])
print("\n== Dependență serială lag 1..60 (chi² 20x20) ==")
print("min p = %.4f la lag %d; p<0.01: %d din 60 (așteptat ~0.6)" % (lagp.min(), res[lagp.argmin()][1], (lagp < 0.01).sum()))
print("Bonferroni: min p * 60 = %.3f" % (lagp.min() * 60))
print("lags cu p<0.05:", [(r[1], round(r[2], 3)) for r in res if r[2] < 0.05])
same = np.array([r[3] for r in res])
print("rata repetare același număr la lag k: min %.4f max %.4f (așteptat 0.05, sd %.4f)" % (same.min(), same.max(), np.sqrt(0.05 * 0.95 / N)))

# difference mod 20
d = (x[1:] - x[:-1]) % K
print("\n== Diferența (x[t]-x[t-1]) mod 20 uniform? p = %.4f" % chisquare(np.bincount(d, minlength=K)).pvalue)
# sums of pairs
s = x[1:] + x[:-1]
# parity / high-low runs
for name, b in [('par/impar', x % 2), ('mic/mare (1-10 vs 11-20)', (x >= 10).astype(int))]:
    runs = 1 + (b[1:] != b[:-1]).sum()
    p1 = b.mean(); mu = 2 * N * p1 * (1 - p1) + 1
    var = 2 * N * p1 * (1 - p1) * (2 * N * p1 * (1 - p1) - N) / (N - 1) if False else (mu - 1) * (mu - 2) / (N - 1)
    z = (runs - mu) / np.sqrt(var)
    print(f"Runs test {name}: z = {z:+.2f}, p = {2 * norm.sf(abs(z)):.4f}")

# drift in time: year x number, month x number
yr = df.dt.dt.year.to_numpy(); mo = df.dt.dt.month.to_numpy() - 1
print("\n== Derivă în timp ==")
print("an × număr p = %.4f" % chi2_contingency(ct(yr - yr.min(), x, yr.max() - yr.min() + 1, K))[1])
print("lună × număr p = %.4f" % chi2_contingency(ct(mo, x, 12, K))[1])
dom = df.dt.dt.day.to_numpy() - 1
print("zi a lunii × număr p = %.4f" % chi2_contingency(ct(dom, x, 31, K))[1])
# position in day equals hour; first draw of day vs prev day's last draw
first = df.groupby('Date').head(1).index.to_numpy()
first = first[first > 0]
print("prima extragere a zilei vs ultima din ziua precedentă p = %.4f" % chi2_contingency(ct(x[first - 1], x[first], K, K))[1])
# same hour previous day(s)
key = df.dt.dt.strftime('%H').to_numpy()
# chunked chi2 over time windows: uniformity per 1000-draw block
blocks = [chisquare(np.bincount(x[i:i + 1000], minlength=K)).pvalue for i in range(0, N - 1000, 1000)]
print("blocuri de 1000: p<0.05 în %d din %d (așteptat ~%.1f); KS pe p-values uniform?" % (sum(p < 0.05 for p in blocks), len(blocks), 0.05 * len(blocks)))
from scipy.stats import kstest
print("  KS p = %.4f" % kstest(blocks, 'uniform').pvalue)

# Triplets: 2-context -> next (400 x 20)
t = np.zeros((K * K, K)); np.add.at(t, (x[:-2] * K + x[1:-1], x[2:]), 1)
t = t[t.sum(1) > 0]
print("\n(x[t-2],x[t-1]) → x[t] p = %.4f" % chi2_contingency(t)[1])
# gap distribution: geometric?
gaps = []
last = {}
for i, v in enumerate(x):
    if v in last: gaps.append(i - last[v])
    last[v] = i
gaps = np.array(gaps)
obs = np.bincount(np.minimum(gaps, 60), minlength=61)[1:]
exp = np.array([0.05 * 0.95 ** (g - 1) for g in range(1, 60)] + [0.95 ** 59]) * len(gaps)
print("distribuția întârzierilor = geometrică? p = %.4f" % chisquare(obs, exp).pvalue)
