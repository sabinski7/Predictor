"""Explorare 28b: cele 10 numere ale combinațiilor complete (combinatii.csv) spun ceva despre Numerone-ul următor?
Rulare din rădăcina repo-ului: python cercetare/explorare28b_combinatii.py"""
import numpy as np, pandas as pd
from scipy.stats import binomtest

c = pd.read_csv("combinatii.csv").sort_values("slot").reset_index(drop=True)
nums = c[[f"n{i}" for i in range(1, 11)]].to_numpy(); m = c.numerone.to_numpy()
print(f"{len(c)} concursuri, {c.slot.min()} – {c.slot.max()}")
for lag in (1, 2, 17):
    ok = np.array([m[t] in nums[t - lag] for t in range(lag, len(c))])
    print(f"Numerone printre cele 10 numere de acum {lag} concursuri: {100*ok.mean():.1f}% (hazard 50%), "
          f"p = {binomtest(int(ok.sum()), len(ok), 0.5).pvalue:.2f}")
for k in (1, 3, 5, 17):
    h = []
    for t in range(k, len(c)):
        top = np.argsort(-np.bincount(nums[t - k:t].ravel(), minlength=21)[1:], kind="stable")[:11] + 1
        h.append(m[t] in top)
    print(f"11 numere cele mai frecvente în ultimele {k} combinații: {100*np.mean(h):.1f}% (hazard 55%)")
