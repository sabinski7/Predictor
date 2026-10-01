"""Faza 9: re-căutarea celei mai bune strategii după FIECARE extragere.
La fiecare tură t: dintre ~40 de strategii, o aleg pe cea cu cele mai multe nimeriri în ultimele W ture
și o folosesc pentru tura t. Comparat cu șansa pură (K/20)."""
import numpy as np, pandas as pd
from scipy.stats import binomtest

df = pd.read_csv('history.csv')
df['dt'] = pd.to_datetime(df.Date + ' ' + df.Time)
df = df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
x = df['Drawn Number'].to_numpy() - 1
N = len(x); K = 20; KK = 9
hours = df.dt.dt.hour.to_numpy()
rng = np.random.default_rng(0)

csum = np.zeros((N + 1, K)); csum[1:] = np.cumsum(np.eye(K)[x], axis=0)
last_seen = np.full(K, -1); gaps = np.zeros((N, K))
m1 = np.zeros((K, K)); M1 = np.zeros((N, K)); hr = np.zeros((24, K)); HR = np.zeros((N, K))
for t in range(N):
    gaps[t] = t - last_seen
    if t: M1[t] = m1[x[t - 1]]
    HR[t] = hr[hours[t]]
    last_seen[x[t]] = t
    if t: m1[x[t - 1], x[t]] += 1
    hr[hours[t], x[t]] += 1

start = 2000
T = np.arange(start, N); y = x[T]
scores = {}
for W in [5, 10, 17, 34, 51, 85, 170, 340, 680, 1700]:
    c = csum[T] - csum[T - W]
    scores[f"calde {W}"] = c; scores[f"reci {W}"] = -c
scores["calde tot"] = csum[T]; scores["reci tot"] = -csum[T]
scores["întârziate"] = gaps[T]; scores["recente"] = -gaps[T]
scores["Markov cald"] = M1[T]; scores["Markov rece"] = -M1[T]
scores["oră caldă"] = HR[T]; scores["oră rece"] = -HR[T]
for L in range(1, 21):                      # „numărul de acum L ture + c” și vecinii lui
    s = np.zeros((len(T), K))
    for c in range(-4, 5):
        s[np.arange(len(T)), (x[T - L] + c) % K] += 1 - abs(c) / 5
    scores[f"acum {L} ±4"] = s
names = list(scores)
tie = rng.random((len(T), K)) * 1e-6
H = np.stack([(np.argsort(-(scores[n] + tie), 1)[:, :KK] == y[:, None]).any(1) for n in names], 1)  # (T, strategii)

print(f"{len(names)} strategii, top-{KK}, șansa pură {100 * KK / K:.0f}%, {len(T):,} ture\n")
cs = np.vstack([np.zeros(len(names)), np.cumsum(H, 0)])
for W in [17, 51, 119, 500, 2000]:
    picks = []
    for i in range(W, len(T)):
        recent = cs[i] - cs[i - W]
        best = np.flatnonzero(recent == recent.max())
        picks.append(H[i, rng.choice(best)])
    picks = np.array(picks)
    p = binomtest(int(picks.sum()), len(picks), KK / K, alternative='greater').pvalue
    print(f"Re-aleg după fiecare tură strategia cea mai bună din ultimele {W:>4} ture: "
          f"{100 * picks.mean():.2f}%  (p = {p:.2f})")
best_fixed = H.mean(0)
print(f"\nCea mai bună strategie fixă, aleasă după ce știm tot (trișând): {100 * best_fixed.max():.2f}% ({names[best_fixed.argmax()]})")
print(f"Media tuturor strategiilor: {100 * best_fixed.mean():.2f}%")
