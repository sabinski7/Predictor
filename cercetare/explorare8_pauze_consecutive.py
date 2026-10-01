"""Faza 8: legătura dintre pauze consecutive (ansamblu vs. serii aleatoare prin același mecanism).
Rulare: python3 explorare8_pauze_consecutive.py <director cu Predictor.ipynb și history.csv> <nr. simulări>"""
import json, os, contextlib, io, numpy as np, matplotlib, sys, time
matplotlib.use('Agg')
run = sys.argv[1]; SIMS = int(sys.argv[2])
nb = json.load(open(f'{run}/Predictor.ipynb'))
code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
os.chdir(run)
g = {'display': lambda *a, **k: None}
with contextlib.redirect_stdout(io.StringIO()):
    for src in code[:7]: exec(src, g)
walk_forward, fit_weights, MODELS = g['walk_forward'], g['fit_weights'], g['MODELS']
hours, N, cal, NEXT = g['hours'], g['N'], g['cal'], g['NEXT_SLOT']
full = np.arange(2000, N)
def corr_for(x):
    P = walk_forward(x, hours, NEXT.hour)
    L = np.stack([P[m][np.arange(N), x] for m in MODELS], axis=1)
    w = fit_weights(L[cal])
    Pe = sum(wi * P[m] for wi, m in zip(w, MODELS))
    order = np.argsort(-Pe[full], axis=1, kind='stable')
    pos = np.argmax(order == x[full][:, None], axis=1)
    out = []
    for K in (7, 8, 9, 10):
        idx = np.flatnonzero(pos < K); gp = np.diff(idx) - 1
        out.append(np.corrcoef(gp[:-1], gp[1:])[0, 1])
    return out
real = corr_for(g['x'])
rng = np.random.default_rng(7); sims = []; t0 = time.time()
for s in range(SIMS):
    sims.append(corr_for(rng.integers(0, 20, N)))
sims = np.array(sims)
print(f"{SIMS} serii aleatoare, {time.time()-t0:.0f}s")
for j, K in enumerate((7, 8, 9, 10)):
    print(f"K={K}: real r={real[j]:+.4f} | aleator: media {sims[:, j].mean():+.4f}, "
          f"sd {sims[:, j].std():.4f}, p(|r_aleator - medie| ≥ |real - medie|) = "
          f"{(np.sum(np.abs(sims[:, j]-sims[:, j].mean()) >= abs(real[j]-sims[:, j].mean()))+1)/(SIMS+1):.3f}")
