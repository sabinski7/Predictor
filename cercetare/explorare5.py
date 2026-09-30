"""Faza 5: rețea neuronală, vecini apropiați (kNN) și potrivirea celei mai lungi secvențe.
Antrenare pe trecut, alegerea parametrilor pe validare, verificare pe ultimele 3.000 de extrageri."""
import numpy as np, pandas as pd, time
from scipy.stats import binomtest, ttest_1samp
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('history.csv')
df['dt'] = pd.to_datetime(df.Date + ' ' + df.Time)
df = df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
x = df['Drawn Number'].to_numpy() - 1
N = len(x); K = 20; TOPK = 7
hours = df.dt.dt.hour.to_numpy(); dows = df.dt.dt.dayofweek.to_numpy()

LAGS = 20
T = np.arange(LAGS, N)
X = np.hstack([np.eye(K)[x[T - l]] for l in range(1, LAGS + 1)]
              + [np.eye(24)[hours[T]][:, 7:24], np.eye(7)[dows[T]]]).astype(np.float32)
y = x[T]
n = len(T)
te = np.arange(n - 3000, n); va = np.arange(n - 6000, n - 3000); tr = np.arange(0, n - 6000)

def score(P, idx):
    ll = -np.log(np.clip(P[np.arange(len(idx)), y[idx]], 1e-12, 1))
    hit = (np.argsort(-P, 1)[:, :TOPK] == y[idx][:, None]).any(1)
    return ll, hit

def report(name, P):
    ll, hit = score(P, te)
    p_ll = ttest_1samp(np.log(K) - ll, 0, alternative='greater').pvalue
    p_hit = binomtest(hit.sum(), len(hit), TOPK / K, alternative='greater').pvalue
    print(f"{name:<44} logloss {ll.mean():.4f} (Δ {np.log(K) - ll.mean():+.4f}, p={p_ll:.3f})  "
          f"top-7 {100 * hit.mean():.2f}% (p={p_hit:.3f})")

print(f"Test pe ultimele 3.000 extrageri; șansa pură top-7 = 35%, log-loss uniform = {np.log(K):.4f}\n")

# ---- 1. Rețea neuronală
t0 = time.time(); best = None
for hidden in [(32,), (128,), (128, 64)]:
    for alpha in [1e-3, 1e-1, 1.0]:
        m = MLPClassifier(hidden, alpha=alpha, max_iter=60, early_stopping=True, random_state=0).fit(X[tr], y[tr])
        ll = score(m.predict_proba(X[va]), va)[0].mean()
        print(f"  MLP {str(hidden):<10} alpha={alpha:<6} valid logloss {ll:.4f}")
        if best is None or ll < best[0]: best = (ll, hidden, alpha)
m = MLPClassifier(best[1], alpha=best[2], max_iter=60, early_stopping=True, random_state=0).fit(X[np.r_[tr, va]], y[np.r_[tr, va]])
report(f"Rețea neuronală {best[1]} alpha={best[2]}", m.predict_proba(X[te]))
print(f"  ({time.time() - t0:.0f}s)\n")

# ---- 2. kNN: situațiile cele mai asemănătoare din trecut
def knn_probs(W, k, idx, pool):
    Z = np.hstack([np.eye(K)[x[T - l]] * (0.8 ** (l - 1)) for l in range(1, W + 1)])
    nn = NearestNeighbors(n_neighbors=k).fit(Z[pool])
    _, nb = nn.kneighbors(Z[idx])
    P = np.full((len(idx), K), 1.0)                      # prior Laplace
    np.add.at(P, (np.repeat(np.arange(len(idx)), k), y[pool][nb].ravel()), 1)
    return P / P.sum(1, keepdims=True)
best = None
for W in [3, 5, 10]:
    for k in [50, 200, 1000]:
        ll = score(knn_probs(W, k, va, tr), va)[0].mean()
        print(f"  kNN fereastră={W:<3} vecini={k:<5} valid logloss {ll:.4f}")
        if best is None or ll < best[0]: best = (ll, W, k)
report(f"kNN (fereastră {best[1]}, {best[2]} vecini)", knn_probs(best[1], best[2], te, np.r_[tr, va]))
print()

# ---- 3. Cea mai lungă secvență care se repetă
def longest_match_probs(idx, min_len=2, max_len=12, prior=20.0):
    P = np.empty((len(idx), K))
    s = ''.join(chr(65 + v) for v in x)
    for r, i in enumerate(idx):
        t = T[i]; hist = s[:t]
        counts = np.zeros(K)
        for L in range(max_len, min_len - 1, -1):
            pat = hist[-L:]
            pos = [j for j in range(len(hist) - L) if hist.startswith(pat, j)] if L >= 6 else None
            if pos is None:
                pos = []; j = hist.find(pat)
                while j != -1 and j < len(hist) - L:
                    pos.append(j); j = hist.find(pat, j + 1)
            if len(pos) >= 3:
                for j in pos: counts[ord(hist[j + L]) - 65] += 1
                break
        P[r] = (counts + prior / K) / (counts.sum() + prior)
    return P
t0 = time.time()
report("Cea mai lungă secvență repetată", longest_match_probs(te))
print(f"  ({time.time() - t0:.0f}s)")
