"""Faza 11: detectorul de schimbare (secțiunea 9c) rulat retroactiv, săptămânal, pe tot istoricul.
Rulare: python3 explorare11_detector_retro.py <director cu Predictor.ipynb și history.csv>"""
import json, os, sys, contextlib, io
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
from scipy.stats import chisquare, binomtest, chi2_contingency, ttest_1samp

run = sys.argv[1]
nb = json.load(open(f'{run}/Predictor.ipynb'))
code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
os.chdir(run); g = {'display': lambda *a, **k: None}
with contextlib.redirect_stdout(io.StringIO()):
    for src in code[:7]: exec(src, g)
x, P, P_ens, N, MODELS, hours, df = g['x'], g['P'], g['P_ens'], g['N'], g['MODELS'], g['hours'], g['df']
K, KK = 20, 9
order = np.argsort(-P_ens[:N], axis=1, kind='stable')
ens_hit = np.argmax(order == x[:, None], axis=1) < KK
gain = {m: np.log(K) + np.log(P[m][np.arange(N), x]) for m in MODELS[1:]}

def ct(a, b, na, nb_):
    t = np.zeros((na, nb_)); np.add.at(t, (a, b), 1); return t[t.sum(1) > 0]

def checks(t_end):
    out = []
    for W in (500, 1700):
        s = slice(t_end - W, t_end); xs = x[s]
        out.append((W, "uniformitate", chisquare(np.bincount(xs, minlength=K)).pvalue))
        rep = int((x[t_end - W:t_end] == x[t_end - W - 1:t_end - 1]).sum())
        out.append((W, "repetă anteriorul", binomtest(rep, W, 1 / K).pvalue))
        diff = (x[t_end - W:t_end] - x[t_end - W - 1:t_end - 1]) % K
        out.append((W, "diferențe consecutive", chisquare(np.bincount(diff, minlength=K)).pvalue))
        if W >= 1700:
            out.append((W, "oră × număr", chi2_contingency(ct(hours[s], xs, 24, K))[1]))
        best = max(gain, key=lambda m: gain[m][s].mean())
        out.append((W, f"cel mai bun model ({best})",
                    min(1.0, ttest_1samp(gain[best][s], 0, alternative="greater").pvalue * (len(MODELS) - 1))))
        out.append((W, "ansamblu top-9 bate șansa",
                    binomtest(int(ens_hit[s].sum()), W, KK / K, alternative="greater").pvalue))
    return out

weeks = np.arange(2000 + 1700, N + 1, 119)          # o verificare pe săptămână (119 ture)
rows = []
for t in weeks:
    c = checks(t); m = len(c)
    pc = [(W, name, min(1.0, p * m)) for W, name, p in c]
    worst = min(pc, key=lambda z: z[2])
    rows.append({"data": df["dt"].iloc[t - 1], "p_corectat": worst[2], "fereastra": worst[0], "verificare": worst[1]})
R = pd.DataFrame(rows)
R["stare"] = np.where(R.p_corectat < 0.01, "🚨", np.where(R.p_corectat < 0.05, "🟡", "✅"))
print(f"{len(R)} verificări săptămânale, {R.data.iloc[0]:%Y-%m-%d} → {R.data.iloc[-1]:%Y-%m-%d}")
print(R.stare.value_counts().to_string())
print("\nToate alarmele (🟡 și 🚨):")
for _, r in R[R.stare != "✅"].iterrows():
    print(f"  {r.stare} {r.data:%Y-%m-%d}  p corectat = {r.p_corectat:.4f}  ({r.verificare}, ultimele {r.fereastra})")

# Cât de des ar suna detectorul pe serii PERFECT aleatoare? (aceeași procedură, 30 de istorii simulate)
rng = np.random.default_rng(0)
fa_y, fa_r = [], []
x_real = x.copy()
for sim in range(30):
    x = rng.integers(0, K, N)
    order_s = np.argsort(-P_ens[:N], axis=1, kind='stable')     # propuneri fixe (independente de seria simulată)
    ens_hit = np.argmax(order_s == x[:, None], axis=1) < KK
    gain = {m: np.log(K) + np.log(P[m][np.arange(N), x]) for m in MODELS[1:]}
    st = []
    for t in weeks:
        c = checks(t); m = len(c)
        st.append(min(min(1.0, p * m) for _, _, p in c))
    st = np.array(st)
    fa_y.append((st < 0.05).mean()); fa_r.append((st < 0.01).mean())
print(f"\nPe 30 de istorii simulate perfect aleatoare: 🟡 sau 🚨 în {100*np.mean(fa_y):.1f}% din săptămâni, "
      f"🚨 în {100*np.mean(fa_r):.1f}% din săptămâni")
print(f"Pe istoricul tău real:                      🟡 sau 🚨 în {100*(R.stare != '✅').mean():.1f}% din săptămâni, "
      f"🚨 în {100*(R.stare == '🚨').mean():.1f}% din săptămâni")
