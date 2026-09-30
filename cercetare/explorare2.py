"""Faza 2: modele ML cu trăsături bogate, antrenare / validare / test cronologic."""
import numpy as np, pandas as pd, time
from scipy.stats import binomtest, ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

df = pd.read_csv('history.csv')
df['dt'] = pd.to_datetime(df.Date + ' ' + df.Time)
df = df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
x = df['Drawn Number'].to_numpy() - 1
N = len(x); K = 20
hours = df.dt.dt.hour.to_numpy(); dows = df.dt.dt.dayofweek.to_numpy()

# ---- features for predicting x[t] using x[:t]
LAGS = [1, 2, 3, 4, 5, 17, 34]
WINS = [17, 85, 340, 1700]
start = 1700
rows = []
csum = np.zeros((N + 1, K)); csum[1:] = np.cumsum(np.eye(K)[x], axis=0)
last_seen = np.full(K, -1); gaps = np.zeros((N, K))
for t in range(N):
    gaps[t] = t - last_seen
    last_seen[x[t]] = t
T = np.arange(start, N)
feats = []
for L in LAGS:
    feats.append(np.eye(K)[x[T - L]])
feats.append(np.eye(24)[hours[T]][:, 7:24])
feats.append(np.eye(7)[dows[T]])
feats.append(np.log1p(gaps[T]))
for W in WINS:
    feats.append((csum[T] - csum[T - W]) / W * K - 1)       # relative deviation
X = np.hstack(feats).astype(np.float32)
y = x[T]
print("X shape", X.shape)

n = len(T)
te = np.arange(n - 3000, n); va = np.arange(n - 6000, n - 3000); tr = np.arange(0, n - 6000)

def report(name, P, idx=te):
    ll = -np.log(np.clip(P[np.arange(len(idx)), y[idx]], 1e-12, 1))
    top7 = np.argsort(-P, axis=1)[:, :7]
    hit = (top7 == y[idx][:, None]).any(1)
    p_ll = ttest_1samp(np.log(K) - ll, 0, alternative='greater').pvalue
    p_hit = binomtest(hit.sum(), len(hit), 0.35, alternative='greater').pvalue
    print(f"{name:<34} logloss {ll.mean():.4f} (Δ {np.log(K) - ll.mean():+.4f}, p={p_ll:.3f})  top7 {100 * hit.mean():.2f}% (p={p_hit:.3f})")
    return hit.mean()

report("Uniform", np.full((3000, K), 1 / K))

# ---- Logistic regression, tune C on validation
sc = StandardScaler().fit(X[tr])
Xs = sc.transform(X)
best = None
for C in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]:
    m = LogisticRegression(C=C, max_iter=300).fit(Xs[tr], y[tr])
    Pv = m.predict_proba(Xs[va])
    ll = -np.log(Pv[np.arange(len(va)), y[va]]).mean()
    print(f"  LR C={C:<7} valid logloss {ll:.4f}")
    if best is None or ll < best[0]: best = (ll, C)
C = best[1]
m = LogisticRegression(C=C, max_iter=300).fit(Xs[np.r_[tr, va]], y[np.r_[tr, va]])
report(f"Regresie logistică (C={C})", m.predict_proba(Xs[te]))

# ---- LightGBM with early stopping on validation
t0 = time.time()
params = dict(objective='multiclass', num_class=K, learning_rate=0.03, num_leaves=15, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=4)
dtr = lgb.Dataset(X[tr], y[tr]); dva = lgb.Dataset(X[va], y[va])
b = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
print(f"  LightGBM best_iter = {b.best_iteration}  ({time.time() - t0:.0f}s)")
b2 = lgb.train(params, lgb.Dataset(X[np.r_[tr, va]], y[np.r_[tr, va]]), max(b.best_iteration, 1))
report("LightGBM (gradient boosting)", b2.predict(X[te]))

# ---- same-hour-yesterday specific signal check
for L in [17, 34, 119]:
    same = (x[L:] == x[:-L]).mean()
    print(f"P(x[t]==x[t-{L}]) = {same:.4f} (așteptat 0.0500)")
