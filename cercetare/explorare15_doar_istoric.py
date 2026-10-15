"""Explorare 15: ce n-am încercat încă, doar pe history.csv.

A. Markov de ordin variabil (0–6) cu backoff bayesian (stil PPM / context tree), walk-forward
B. Model ascuns Markov (HMM, 2–4 stări / „regimuri”), antrenat pe 70%, testat pe 30%
C. Interacțiuni oră × lag (fiecare oră, lag 1–17 și 34, 51, 119)
D. Baterie tip NIST pe secvență: entropie aproximativă, triplete suprapuse, sume cumulative, cea mai lungă serie
Verdict: log-loss față de uniform (ln 20 = 2.9957) și rata top-9 pe ultimele 10.000 de ture față de 45%.
"""
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv("history.csv")
x = df["Drawn Number"].to_numpy() - 1
hrs = pd.to_datetime(df["Time"], format="%H:%M").dt.hour.to_numpy()
N, K = len(x), 20
TEST = 10000
T0 = N - TEST
LN20 = np.log(K)
print(f"Istoric: {N} extrageri, test pe ultimele {TEST}\n")


def score(P, name):
    idx = np.arange(T0, N)
    ll = -np.mean(np.log(P[idx, x[idx]]))
    order = np.argsort(-P[idx], axis=1, kind="stable")
    rank = np.argmax(order == x[idx][:, None], axis=1)
    h1, h9 = np.mean(rank < 1), np.mean(rank < 9)
    p9 = stats.binomtest(int((rank < 9).sum()), TEST, 0.45, alternative="greater").pvalue
    print(f"  {name:38s} log-loss {ll:.4f} (Δ {ll-LN20:+.4f})  top1 {100*h1:5.2f}%  top9 {100*h9:5.2f}%  p={p9:.3f}")
    return ll, h9


# ---------- A. Markov ordin variabil ----------
print("A. Markov de ordin variabil (Dirichlet + backoff)")
def vom(maxo, alpha):
    P = np.full((N, K), 1 / K)
    tabs = [dict() for _ in range(maxo + 1)]
    for t in range(N):
        p = np.full(K, 1 / K)
        for o in range(0, maxo + 1):
            if t < o:
                break
            key = tuple(x[t - o:t])
            c = tabs[o].get(key)
            if c is not None:
                p = (c + alpha * K * p) / (c.sum() + alpha * K)
        P[t] = p
        for o in range(0, maxo + 1):
            if t < o:
                break
            key = tuple(x[t - o:t])
            if key not in tabs[o]:
                tabs[o][key] = np.zeros(K)
            tabs[o][key][x[t]] += 1
    return P

res = {}
for maxo in [1, 2, 3, 4, 6]:
    for alpha in [1.0, 10.0]:
        res[(maxo, alpha)] = score(vom(maxo, alpha), f"ordin ≤{maxo}, α={alpha}")

# ---------- B. HMM categorial ----------
print("\nB. HMM (regimuri ascunse), antrenat pe 70%, filtrat walk-forward pe test")
def hmm_fit(seq, S, iters=60, seed=0):
    r = np.random.default_rng(seed)
    A = np.full((S, S), 0.05 / max(S - 1, 1)); np.fill_diagonal(A, 0.95)
    B = r.dirichlet(np.ones(K) * 50, size=S)
    pi = np.full(S, 1 / S)
    T = len(seq)
    for _ in range(iters):
        al = np.zeros((T, S)); c = np.zeros(T)
        a = pi * B[:, seq[0]]; c[0] = a.sum(); al[0] = a / c[0]
        for t in range(1, T):
            a = (al[t - 1] @ A) * B[:, seq[t]]; c[t] = a.sum(); al[t] = a / c[t]
        be = np.ones((T, S))
        for t in range(T - 2, -1, -1):
            be[t] = (A @ (B[:, seq[t + 1]] * be[t + 1])) / c[t + 1]
        g = al * be
        xi = np.zeros((S, S))
        for t in range(T - 1):
            xi += (al[t][:, None] * A * (B[:, seq[t + 1]] * be[t + 1])[None, :]) / c[t + 1]
        A = xi / xi.sum(1, keepdims=True)
        B = np.array([np.bincount(seq, weights=g[:, s], minlength=K) for s in range(S)]) + 1e-3
        B /= B.sum(1, keepdims=True)
        pi = g[0]
    return A, B, np.log(c).sum()

def hmm_predict(A, B):
    S = len(A)
    P = np.full((N, K), 1 / K)
    f = np.full(S, 1 / S)
    for t in range(N):
        pr = f @ A
        P[t] = pr @ B
        f = pr * B[:, x[t]]; f /= f.sum()
    return P

train = x[:int(0.7 * N)]
for S in [2, 3, 4]:
    A, B, ll = hmm_fit(train, S)
    spread = np.abs(B - 1 / K).max()
    print(f"  {S} stări: persistență diag {np.round(np.diag(A), 3)}, abaterea max. a unei stări de la 5%: {100*spread:.2f} pp")
    score(hmm_predict(A, B), f"HMM {S} stări")

# ---------- C. oră × lag ----------
print("\nC. Oră × lag: P(numărul de acum = numărul de acum L ture), separat pe fiecare oră")
lags = list(range(1, 18)) + [34, 51, 119]
rows = []
for h in range(7, 24):
    for L in lags:
        idx = np.flatnonzero((hrs == h) & (np.arange(N) >= L))
        eq = (x[idx] == x[idx - L]).sum()
        p = stats.binomtest(int(eq), len(idx), 0.05).pvalue
        rows.append((h, L, len(idx), eq / len(idx), p))
cl = pd.DataFrame(rows, columns=["ora", "lag", "n", "rata", "p"])
m = len(cl)
print(f"  {m} teste; prag Bonferroni {0.05/m:.1e}; cel mai mic p = {cl.p.min():.2e}; "
      f"sub 0.05 fără corecție: {(cl.p<0.05).sum()} (așteptat ~{0.05*m:.0f})")
print(cl.nsmallest(5, "p").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# și oră × „același număr ca diferență”: x_t - x_{t-L} mod 20 constant
rows = []
for h in range(7, 24):
    for L in lags:
        idx = np.flatnonzero((hrs == h) & (np.arange(N) >= L))
        d = (x[idx] - x[idx - L]) % K
        obs = np.bincount(d, minlength=K)
        chi, p = stats.chisquare(obs)
        rows.append((h, L, p))
cd = pd.DataFrame(rows, columns=["ora", "lag", "p"])
print(f"  diferențe (x_t − x_t−L) mod 20 pe oră: {len(cd)} teste chi², min p {cd.p.min():.2e}, "
      f"sub 0.05: {(cd.p<0.05).sum()} (așteptat ~{0.05*len(cd):.0f}); KS pe p-uri: p={stats.kstest(cd.p,'uniform').pvalue:.3f}")

# ---------- D. baterie tip NIST ----------
print("\nD. Baterie tip NIST pe secvența de numere")
def apen(seq, m):
    def phi(mm):
        codes = np.zeros(len(seq) - mm + 1, dtype=np.int64)
        for j in range(mm):
            codes = codes * K + seq[j:len(seq) - mm + 1 + j]
        cnt = np.bincount(codes, minlength=K ** mm)
        q = cnt[cnt > 0] / cnt.sum()
        return (q * np.log(q)).sum()
    return phi(m) - phi(m + 1)
# null prin simulare
r = np.random.default_rng(1)
for m_ in [1, 2]:
    a = apen(x, m_)
    null = np.array([apen(r.integers(0, K, N), m_) for _ in range(200)])
    p = 2 * min((null <= a).mean(), (null >= a).mean())
    print(f"  entropie aproximativă m={m_}: {a:.5f} vs nul {null.mean():.5f}±{null.std():.5f}  p≈{max(p,1/200):.3f}")
codes = x[:-2] * 400 + x[1:-1] * 20 + x[2:]
obs = np.bincount(codes, minlength=8000)
null = np.array([np.var(np.bincount(r.integers(0, 8000, len(codes)), minlength=8000)) for _ in range(300)])
v = np.var(obs)
print(f"  triplete suprapuse (8000 celule): varianță {v:.3f} vs nul {null.mean():.3f}±{null.std():.3f}  "
      f"p≈{max(2*min((null<=v).mean(),(null>=v).mean()),1/300):.3f}")
pc, pl = [], []
for k in range(K):
    s = np.where(x == k, 1, -1)
    # suma cumulativă pentru o secvență ±1 cu P(+1)=0.05: centrăm
    z = np.cumsum((x == k) - 0.05)
    zmax = np.abs(z).max() / np.sqrt(N * 0.05 * 0.95)
    pc.append(zmax)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, (x != k).astype(int), 0]) != 0))[::2]
    pl.append(runs.max())
# null pentru sume cumulative și cea mai lungă pauză
nc, nl = [], []
for _ in range(300):
    y = r.integers(0, K, N) == 0
    z = np.cumsum(y - 0.05); nc.append(np.abs(z).max() / np.sqrt(N * 0.05 * 0.95))
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, (~y).astype(int), 0]) != 0))[::2]; nl.append(runs.max())
nc, nl = np.array(nc), np.array(nl)
pcs = [(nc >= c).mean() for c in pc]; pls = [(nl >= l).mean() for l in pl]
print(f"  sume cumulative pe fiecare număr: min p {min(pcs):.3f}, KS uniform p={stats.kstest(pcs,'uniform').pvalue:.3f}")
print(f"  cea mai lungă pauză pe număr: {dict(zip(range(1,21), pl))}")
print(f"     nul: mediană {np.median(nl):.0f}, 95% ≤ {np.percentile(nl,95):.0f};  min p {min(pls):.3f}, KS p={stats.kstest(pls,'uniform').pvalue:.3f}")
