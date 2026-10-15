"""Explorare 16: rețea neuronală secvențială (GRU) pe history.csv.
Intrare: ultimele 68 de numere (4 zile) + ora. Antrenare pe tot ce e înainte de ultimele 10.000 de ture
(ultimii 15% din antrenare pentru oprire timpurie), test pe ultimele 10.000.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import stats
torch.manual_seed(0); np.random.seed(0)
df = pd.read_csv("history.csv")
x = df["Drawn Number"].to_numpy() - 1
h = pd.to_datetime(df["Time"], format="%H:%M").dt.hour.to_numpy() - 7
N, K, W, TEST = len(x), 20, 68, 10000
idx = np.arange(W, N)
X = np.stack([x[i - W:i] for i in idx]); H = h[idx]; Y = x[idx]
T0 = N - TEST - W
cut = int(T0 * 0.85)
tr, va, te = slice(0, cut), slice(cut, T0), slice(T0, None)

class Net(nn.Module):
    def __init__(s, d=32, hid=64):
        super().__init__()
        s.e = nn.Embedding(K, d); s.eh = nn.Embedding(17, 8)
        s.g = nn.GRU(d, hid, batch_first=True); s.drop = nn.Dropout(0.3)
        s.o = nn.Linear(hid + 8, K)
    def forward(s, xs, hs):
        _, z = s.g(s.e(xs))
        return s.o(s.drop(torch.cat([z[-1], s.eh(hs)], 1)))

T = lambda a: torch.tensor(a, dtype=torch.long)
Xt, Ht, Yt = T(X), T(H), T(Y)
net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()
best, best_state, bad = 9, None, 0
for ep in range(30):
    net.train()
    perm = torch.randperm(cut)
    for b in range(0, cut, 256):
        j = perm[b:b + 256]
        opt.zero_grad(); l = lossf(net(Xt[j], Ht[j]), Yt[j]); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        vl = lossf(net(Xt[va], Ht[va]), Yt[va]).item()
    print(f"epoca {ep:2d}: validare log-loss {vl:.4f} (uniform {np.log(K):.4f})")
    if vl < best - 1e-4:
        best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= 4:
            break
net.load_state_dict(best_state); net.eval()
with torch.no_grad():
    P = torch.softmax(net(Xt[te], Ht[te]), 1).numpy()
y = Y[te]
ll = -np.log(P[np.arange(len(y)), y]).mean()
rank = np.argmax(np.argsort(-P, 1, kind="stable") == y[:, None], 1)
h9 = (rank < 9).sum()
print(f"\nTEST ultimele {len(y)}: log-loss {ll:.4f} (Δ {ll-np.log(K):+.4f}), top1 {100*(rank<1).mean():.2f}%, "
      f"top9 {100*h9/len(y):.2f}%  p={stats.binomtest(int(h9), len(y), 0.45, alternative='greater').pvalue:.3f}")
