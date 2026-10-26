"""Explorare 21: căutare largă a celei mai bune strategii cu 11 numere (cele mai scurte pauze = cea mai mare nimerire).
70 de metode de a alege 11 numere, fiecare folosind doar extragerile anterioare:
modelele notebook-ului, frecvențe pe 12 ferestre (calde/reci), recență, aceeași oră în ultimele zile,
excluderea ultimelor extrageri, tranziții Markov, „deplasări” față de numărul de acum L ture,
regulile din top 10 + propuneri, și combinații (vot) ale celor mai bune.
Disciplina: metodele se clasează pe prima jumătate a istoricului; câștigătorul se verifică pe a doua jumătate.
Rulare din rădăcina repo-ului: python cercetare/explorare21_cautare_11.py"""
import sys, json, contextlib, io
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
from scipy.stats import binomtest
sys.path.insert(0, "automat")

K, NN = 11, 20
nb = json.load(open("Predictor.ipynb"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {"display": lambda *a, **k: None}
with contextlib.redirect_stdout(io.StringIO()):
    for src in code[:7]:
        exec(src, g)
x, N = g["x"], g["N"]                      # x: 0..19
hours = g["hours"] if "hours" in g else None
df = pd.read_csv("history.csv")
hrs = pd.to_datetime(df["Time"], format="%H:%M").dt.hour.to_numpy()[:N]
T0 = 2000
idx = np.arange(T0, N)
mid = T0 + (N - T0) // 2
A, B = idx < mid, idx >= mid
tie = np.arange(NN) * 1e-9                 # departajare fixă

S = {}                                      # nume -> matrice scor (N, 20); se joacă top 11
def add(name, M):
    S[name] = np.asarray(M, float)[:N]

# 1) modelele notebook-ului
for nm, M in g["P"].items():
    add(f"model: {nm}", M)
add("model: ansamblu (notebook)", g["P_ens"])

# 2) frecvențe pe ferestre
oh = np.eye(NN)[x]
cs = np.vstack([np.zeros(NN), np.cumsum(oh, 0)])        # cs[t] = numărări în [0, t)
for W in [17, 34, 51, 85, 119, 170, 255, 340, 510, 1000, 2000, 5000]:
    cnt = cs[np.arange(N)] - cs[np.maximum(np.arange(N) - W, 0)]
    add(f"calde, ultimele {W}", cnt); add(f"reci, ultimele {W}", -cnt)
add("calde, tot istoricul", cs[:N]); add("reci, tot istoricul", -cs[:N])

# 3) recență
last = np.full(NN, -1.0); L = np.zeros((N, NN))
for t in range(N):
    L[t] = last; last[x[t]] = t
add("recente (ieșite de curând)", L); add("restante (neieșite de mult)", -L)

# 4) aceeași oră
hcnt = {}
for D in [7, 14, 30, 60, 120, 365]:
    M = np.zeros((N, NN))
    for h in range(7, 24):
        ii = np.flatnonzero(hrs == h)
        c = np.vstack([np.zeros(NN), np.cumsum(oh[ii], 0)])
        k = np.arange(len(ii))
        M[ii] = c[k] - c[np.maximum(k - D, 0)]
    add(f"aceeași oră, calde, {D} zile", M); add(f"aceeași oră, reci, {D} zile", -M)

# 5) excluderea ultimelor m extrageri (restul ordonat după ansamblu)
for m in range(1, 10):
    M = g["P_ens"][:N].copy() * 1e-3
    for j in range(1, m + 1):
        M[np.arange(j, N), x[:N - j]] -= 1
    add(f"fără ultimele {m} extrageri", M)

# 6) tranziții Markov: ce a urmat după numărul precedent
for W in [500, 2000, None]:
    T = np.zeros((NN, NN)); M = np.zeros((N, NN)); hist = []
    for t in range(1, N):
        M[t] = T[x[t - 1]]
        T[x[t - 1], x[t]] += 1; hist.append((x[t - 1], x[t]))
        if W and len(hist) > W:
            a, b = hist.pop(0); T[a, b] -= 1
    add(f"Markov (după numărul precedent), {W or 'tot'}", M)

# 7) deplasări față de numărul de acum L ture; deplasările alese pe prima jumătate
for Lg in [1, 2, 3, 17, 34]:
    t_ = idx[A]; d = (x[t_] - x[t_ - Lg]) % NN
    freq = np.bincount(d, minlength=NN).astype(float)
    M = np.zeros((N, NN))
    for t in range(Lg, N):
        M[t, (x[t - Lg] + np.arange(NN)) % NN] = freq
    add(f"deplasări față de lag {Lg} (învățate pe jumătatea 1)", M)

# rangul numărului ieșit pentru fiecare metodă → nimerire cu 11
y = x[idx]
H = {}
for nm, M in S.items():
    sc = M[idx] + tie
    rank = (sc > sc[np.arange(len(idx)), y][:, None]).sum(1)
    H[nm] = rank < K

# 8) regulile din top 10 + propuneri (strategia actuală)
import pauze
C = pauze.C; NB = len(C); NR = NB * 20
order = np.argsort(-g["P_ens"], axis=1, kind="stable") + 1
xr = x + 1; hit_rules = np.zeros(len(idx), bool)
for b0 in range(T0, N, 500):
    run = np.zeros(NR, np.int32); mx = run.copy(); hits = run.copy()
    for t in range(b0, min(N, b0 + 500)):
        props = order[t, :9]; base = props @ C.T
        if t - b0 >= 50:
            key = np.maximum(mx, run).astype(np.int64) * 100000 - hits
            ii = np.argpartition(key, 10)[:10]; ii = ii[np.argsort(key[ii], kind="stable")]
            rules = [int((base[i // 20] + i % 20 - 1) % 20 + 1) for i in ii]
            nums = pauze.numere_11(rules, props, xr[:t])
        else:
            nums = list(order[t, :K])
        hit_rules[t - T0] = xr[t] in nums
        h = np.arange(NB) * 20 + (xr[t] - base) % 20
        mx = np.maximum(mx, run); run += 1; run[h] = 0; hits[h] += 1
H["reguli top 10 + propuneri (strategia actuală)"] = hit_rules

# 9) combinații: vot între cele mai bune 3/5/10 metode pe jumătatea 1
rankA = sorted(S, key=lambda n: -H[n][A].mean())
for k in (3, 5, 10):
    tot = np.zeros((len(idx), NN))
    for nm in rankA[:k]:
        sc = S[nm][idx] + tie
        tot += np.argsort(np.argsort(-sc, 1), 1)          # rang 0 = cel mai bun
    sc = -tot + tie
    rank = (sc > sc[np.arange(len(idx)), y][:, None]).sum(1)
    H[f"vot între primele {k} metode (jumătatea 1)"] = rank < K
H["11 numere la întâmplare"] = np.random.default_rng(0).random(len(idx)) < 0  # completat mai jos
rng = np.random.default_rng(0)
H["11 numere la întâmplare"] = np.array([v in rng.choice(NN, K, replace=False) for v in y])


def maxpause(h):
    r = m = 0
    for v in h:
        r = 0 if v else r + 1; m = max(m, r)
    return m


rows = []
for nm, h in H.items():
    rows.append((nm, 100 * h[A].mean(), 100 * h[B].mean(), maxpause(h[A]), maxpause(h[B]),
                 binomtest(int(h[B].sum()), int(B.sum()), 0.55).pvalue))
R = pd.DataFrame(rows, columns=["metoda", "jum1 %", "jum2 %", "pauza max jum1", "pauza max jum2", "p jum2"])
R = R.sort_values("jum1 %", ascending=False)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60)
print(f"{len(H)} metode; jumătatea 1: {A.sum():,} ture, jumătatea 2: {B.sum():,} ture; hazard 55%\n")
print("Primele 15 după jumătatea 1 (căutare) și cum se comportă pe jumătatea 2 (verificare):")
print(R.head(15).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print("\nUltimele 5:")
print(R.tail(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(f"\nMedia celor mai bune 10 pe jumătatea 1: {R['jum1 %'].head(10).mean():.2f}% → pe jumătatea 2: {R['jum2 %'].head(10).mean():.2f}%")
print(f"Corelație jumătatea 1 ↔ jumătatea 2 între metode: {np.corrcoef(R['jum1 %'], R['jum2 %'])[0,1]:+.2f}")
best = R.sort_values("jum2 %", ascending=False).iloc[0]
print(f"Cea mai bună pe jumătatea 2 (aleasă după rezultat, deci optimistă): {best.metoda} {best['jum2 %']:.2f}%")
R.to_csv("/tmp/claude-0/-home-user-Predictor/9da01c27-98cd-5380-9d0a-6e35dab68510/scratchpad/e21.csv", index=False)
