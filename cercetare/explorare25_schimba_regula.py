"""Explorare 25: „mergem pe o regulă; când scade, trecem pe una care merge bine acum”.
La fiecare tură joci numărul unei singure reguli. Regula se schimbă când:
  (a) a ratat de m ori la rând, sau
  (b) rata ei pe ultimele W ture a scăzut sub 5%;
și se alege atunci regula cu cele mai multe nimeriri pe ultimele W ture (cea care „merge bine acum”).
Testat pe istoric (reguli pe numerele ieșite n1…n9, 33.000+ ture) și pe jurnalul tău (reguli pe propuneri).
Rulare din rădăcina repo-ului: python cercetare/explorare25_schimba_regula.py"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "automat")
import pauze

C = pauze.C; NB = len(C); NR = NB * 20
h = pd.read_csv("history.csv"); x = h["Drawn Number"].to_numpy(); N = len(x)
LAG = np.stack([np.r_[np.zeros(k, int), x[:-k]] for k in range(1, 10)], 1)


def run(inputs, y, W, trigger, m):
    """întoarce vectorul de nimeriri al strategiei și câte schimbări a făcut."""
    n = len(y); recent = np.zeros(NR, np.int32); buf = []
    cur, miss, switches, out = None, 0, 0, []
    for i in range(n):
        base = inputs[i] @ C.T
        if i >= W:
            bad = cur is None or (miss >= m if trigger == "ratări" else recent[cur] < 0.05 * W)
            if bad:
                cands = np.flatnonzero(recent == recent.max()); cur = int(cands[0]); switches += 1; miss = 0
            pred = (base[cur // 20] + cur % 20 - 1) % 20 + 1
            hit = pred == y[i]; out.append(hit); miss = 0 if hit else miss + 1
        hits = np.arange(NB) * 20 + (y[i] - base) % 20
        recent[hits] += 1; buf.append(hits)
        if len(buf) > W:
            recent[buf.pop(0)] -= 1
    return np.array(out), switches


def show(name, inputs, y):
    print(f"\n{name}")
    print(f"   {'regula de schimbare':<44}{'ture':>8}{'nimerire':>10}{'schimbări':>11}")
    for W in (17, 51, 100, 300):
        for trig, m in [("ratări", 5), ("ratări", 20), ("ratări", 40), ("rată sub 5%", 0)]:
            h_, s = run(inputs, y, W, trig, m)
            lab = f"W={W}, " + (f"după {m} ratări la rând" if trig == "ratări" else "când rata pe W scade sub 5%")
            print(f"   {lab:<44}{len(h_):8,}{100*h_.mean():9.2f}%{s:11,}")


show(f"ISTORIC: reguli pe numerele ieșite, {N-9:,} ture", LAG[9:], x[9:])
lg, _ = pauze.load()
P = np.array([[int(v) for v in p.split()] for p in lg.propuneri])
show(f"JURNALUL TĂU: reguli pe propuneri, {len(lg)} ture", P, lg.rez.to_numpy(int))
print("\nȘansa pură: 5,00%. Pe istoric, intervalul normal din hazard e cam ±0,25 pp; pe jurnal cam ±2 pp.")
