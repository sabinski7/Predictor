"""Faza 6: căutare exhaustivă de reguli aritmetice de tipul
   următorul = (a · x[t-L] + c) mod 20            (L = 1..500, a ∈ {1,-1,3,-3,7,-7,9,-9,...}, c = 0..19)
   următorul = (±x[t-a] ± x[t-b] + c) mod 20      (1 ≤ a < b ≤ 30)
Fiecare regulă prezice UN număr → șansa pură 5%. Regulile se descoperă pe trecut și se verifică pe ultimele 3.000."""
import numpy as np, pandas as pd
from scipy.stats import binom

df = pd.read_csv('history.csv')
df['dt'] = pd.to_datetime(df.Date + ' ' + df.Time)
df = df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
x = df['Drawn Number'].to_numpy() - 1          # 0..19
N = len(x); K = 20
START = 500                                    # ca toate regulile să aibă istoric
T = np.arange(START, N)
y = x[T]
past = T < N - 3000; fut = ~past
n_past, n_fut = past.sum(), fut.sum()

rules = []   # (descriere, rata trecut, rata viitor)
def eval_pred(desc, pred):
    """pred: (len(T),) numărul prezis. Adaugă regula cu ratele de nimerire."""
    hit = pred == y
    rules.append((desc, hit[past].mean(), hit[fut].mean()))

def eval_family(desc_fn, base):
    """base = combinația fără constanta c; testează toate cele 20 de deplasări c deodată."""
    d = (y - base) % K                         # c-ul care ar fi nimerit
    cp = np.bincount(d[past], minlength=K) / n_past
    cf = np.bincount(d[fut], minlength=K) / n_fut
    for c in range(K):
        rules.append((desc_fn(c), cp[c], cf[c]))

# 1. reguli simple a·x[t-L] + c
for L in range(1, 501):
    for a in [1, -1, 3, -3, 7, -7, 9, -9, 11, -11, 13, -13, 17, -17, 19, 2, -2, 5, 4, 10]:
        eval_family(lambda c, L=L, a=a: f"{a:+d}·(acum {L}) {c:+d}", (a * x[T - L]) % K)
# 2. combinații de două extrageri
for A in range(1, 31):
    for B in range(A + 1, 31):
        for sa, sb in [(1, 1), (1, -1), (-1, 1)]:
            eval_family(lambda c, A=A, B=B, sa=sa, sb=sb: f"{'+' if sa > 0 else '-'}(acum {A}) {'+' if sb > 0 else '-'}(acum {B}) {c:+d}",
                        (sa * x[T - A] + sb * x[T - B]) % K)
# 3. aceeași oră din zilele trecute (+c), dacă programul e complet
for d in range(1, 31):
    eval_family(lambda c, d=d: f"aceeași oră acum {d} zile {c:+d}", x[T - 17 * d])

R = pd.DataFrame(rules, columns=["regula", "trecut", "viitor"])
print(f"Reguli testate: {len(R):,}   (trecut: {n_past:,} extrageri, viitor: {n_fut:,}; șansa pură 5,00%)\n")

top = R.sort_values("trecut", ascending=False).head(15)
print("Cele mai bune 15 reguli pe TRECUT și cum s-au descurcat în VIITOR:")
for _, r in top.iterrows():
    print(f"  {r.regula:<36} trecut {100 * r.trecut:5.2f}%   viitor {100 * r.viitor:5.2f}%")

best_past = R.trecut.max()
# probabilitatea ca MĂCAR UNA din N reguli aleatoare să ajungă la best_past din pur noroc
p_one = binom.sf(round(best_past * n_past) - 1, n_past, 1 / K)
p_any = 1 - (1 - p_one) ** len(R)
print(f"\nCea mai bună regulă pe trecut: {100 * best_past:.2f}%. Șansa ca, din {len(R):,} reguli fără sens, "
      f"măcar una să ajungă aici din noroc: {100 * p_any:.0f}%")
print(f"Media în viitor a celor mai bune 100 de reguli pe trecut: "
      f"{100 * R.sort_values('trecut', ascending=False).head(100).viitor.mean():.2f}%")
print(f"Corelație trecut ↔ viitor pe toate regulile: {np.corrcoef(R.trecut, R.viitor)[0, 1]:+.3f}")
print(f"Cea mai bună regulă din toate (trecut + viitor combinat): "
      f"{100 * ((R.trecut * n_past + R.viitor * n_fut) / (n_past + n_fut)).max():.2f}%")

# regulile exacte pe care le-a menționat utilizatorul
print("\nExemplele tale:")
for q in ["+1·(acum 1) +1", "+1·(acum 10) +1", "+1·(acum 1) +0", "+1·(acum 17) +0"]:
    r = R[R.regula == q].iloc[0]
    print(f"  {q:<22} trecut {100 * r.trecut:5.2f}%   viitor {100 * r.viitor:5.2f}%")
