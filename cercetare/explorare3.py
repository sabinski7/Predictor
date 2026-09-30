"""Faza 3: zeci de strategii „populare”; selecție pe trecut, verificare pe viitor."""
import numpy as np, pandas as pd
from scipy.stats import binomtest

df = pd.read_csv('history.csv')
df['dt'] = pd.to_datetime(df.Date + ' ' + df.Time)
df = df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
x = df['Drawn Number'].to_numpy() - 1
N = len(x); K = 20; KK = 7
hours = df.dt.dt.hour.to_numpy()
rng = np.random.default_rng(0)
tie = rng.random((N, K)) * 1e-6           # rupere aleatoare a egalităților

csum = np.zeros((N + 1, K)); csum[1:] = np.cumsum(np.eye(K)[x], axis=0)
last_seen = np.full(K, -1); gaps = np.zeros((N, K))
m1 = np.zeros((K, K)); M1 = np.zeros((N, K))
hr = np.zeros((24, K)); HR = np.zeros((N, K))
for t in range(N):
    gaps[t] = t - last_seen
    if t: M1[t] = m1[x[t - 1]]
    HR[t] = hr[hours[t]]
    last_seen[x[t]] = t
    if t: m1[x[t - 1], x[t]] += 1
    hr[hours[t], x[t]] += 1

start = 2000
T = np.arange(start, N)
scores = {}
for W in [5, 10, 17, 34, 51, 85, 170, 340, 680, 1700]:
    c = csum[T] - csum[T - W]
    scores[f"cele mai calde, ultimele {W}"] = c
    scores[f"cele mai reci, ultimele {W}"] = -c
scores["cele mai calde, tot istoricul"] = csum[T]
scores["cele mai reci, tot istoricul"] = -csum[T]
scores["cele mai întârziate (gap mare)"] = gaps[T]
scores["cele mai recente (gap mic)"] = -gaps[T]
scores["Markov-1 cald"] = M1[T]
scores["Markov-1 rece"] = -M1[T]
scores["oră din zi cald"] = HR[T]
scores["oră din zi rece"] = -HR[T]
for L in [17, 34]:
    s = np.zeros((len(T), K)); s[np.arange(len(T)), x[T - L]] = 1
    for j in range(1, 7): s[np.arange(len(T)), x[T - L - j]] += 0.5 ** j
    scores[f"vecinii de acum {L} extrageri (ieri/alaltăieri)"] = s

y = x[T]
sel = np.arange(0, len(T) - 3000)       # perioada pe care „alegem” cea mai bună
te = np.arange(len(T) - 3000, len(T))   # viitorul nevăzut
res = []
for name, s in scores.items():
    top = np.argsort(-(s + tie[T]), axis=1)[:, :KK]
    hit = (top == y[:, None]).any(1)
    res.append((name, hit[sel].mean(), hit[te].mean(), hit[te].sum()))
res.sort(key=lambda r: -r[1])
print(f"{'Strategie':<48} {'trecut':>8} {'viitor':>8}")
for name, a, b, _ in res:
    print(f"{name:<48} {100 * a:7.2f}% {100 * b:7.2f}%")
best = res[0]
print(f"\nCea mai bună pe trecut: {best[0]} → {100 * best[1]:.2f}% ; pe viitor: {100 * best[2]:.2f}% "
      f"(p vs 35% = {binomtest(best[3], 3000, 0.35, alternative='greater').pvalue:.3f})")
print("Media tuturor strategiilor pe viitor: %.2f%%" % (100 * np.mean([r[2] for r in res])))
print("Corelație trecut ↔ viitor între strategii: %.2f" % np.corrcoef([r[1] for r in res], [r[2] for r in res])[0, 1])
print("Nr. strategii:", len(res))
