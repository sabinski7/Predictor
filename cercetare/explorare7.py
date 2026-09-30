"""Faza 7: „numerele rămase în urmă trebuie să recupereze”? (legea numerelor mari vs. eroarea jucătorului)"""
import numpy as np, pandas as pd
from scipy.stats import binomtest

df = pd.read_csv('history.csv')
df['dt'] = pd.to_datetime(df.Date + ' ' + df.Time)
df = df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
x = df['Drawn Number'].to_numpy() - 1
N = len(x); K = 20
csum = np.zeros((N + 1, K)); csum[1:] = np.cumsum(np.eye(K)[x], axis=0)

print("1) Ce se egalează de fapt cu timpul?\n")
print(f"{'extrageri':>10} | {'diferența max–min (bucăți)':>27} | {'diferența max–min (% din total)':>31}")
for n in [100, 300, 1000, 3000, 10000, 30000]:
    c = csum[n]
    print(f"{n:>10,} | {c.max() - c.min():>27.0f} | {100 * (c.max() - c.min()) / n:>30.2f}%")

print("\n2) Numerele rămase în urmă recuperează? (ce se întâmplă în următoarele 100 de extrageri)\n")
H = 100
T = np.arange(1000, N - H, H)         # ferestre care nu se suprapun
for name, W in [("ultimele 100", 100), ("ultimele 500", 500), ("tot istoricul", None)]:
    rows = []
    for t in T:
        c = csum[t] - (csum[t - W] if W else 0)
        order = np.argsort(c + np.random.default_rng(t).random(K) * 1e-6)
        nxt = csum[t + H] - csum[t]
        rows.append((nxt[order[:7]].sum(), nxt[order[-7:]].sum()))
    rows = np.array(rows)
    cold, hot = rows[:, 0].sum(), rows[:, 1].sum()
    tot = len(T) * H
    p = binomtest(int(cold), tot, 7 / K).pvalue
    print(f"  după {name:<14}: cele mai RECI 7 au ieșit {100 * cold / tot:5.2f}%, "
          f"cele mai CALDE 7 au ieșit {100 * hot / tot:5.2f}%  (așteptat 35,00%; p reci = {p:.2f})")
