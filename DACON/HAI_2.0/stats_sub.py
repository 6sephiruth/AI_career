from TaPR_pkg import etapr
from pathlib import Path
from itertools import groupby

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless operation
from matplotlib import pyplot as plt

WIN = 33
TH = 0.07

def nanth(s, th):
    if math.isnan(s):   return 0.0
    elif s >= th:       return 1.0
    else:               return 0.0

# plot anomaly
def plt_anom(xs, att, piece=2, filename='anomaly', thresh=None, tapr=None):
    l = xs.shape[0]
    chunk = l // piece
    plt.title("validation")
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].plot(xticks, xs[L:R])
        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            axs[i].plot(xticks, att[L:R] * peak * 0.3)
        if thresh!=None:
            axs[i].axhline(y=thresh, color='r')
    plt.savefig(filename)

solpath = sorted([x for x in Path("archives/submissions/").glob("*.csv")])
smpath = sorted([x for x in Path("archives/smoothed/").glob("*.csv")])

bestsol = solpath[-2]
bdf = pd.read_csv(bestsol)

bestsm = smpath[-1]
smdf = pd.read_csv(bestsm)

b_ts = bdf.time
b_attk = bdf.attack

s_ts = smdf.time
s_attk = smdf.attack

mavg = b_attk.rolling(WIN, min_periods=1).mean().apply(lambda s: nanth(s, TH))

plt_anom(b_attk, mavg, piece=5, filename="test")

d1 = b_attk.shape[0] - (b_attk == mavg).value_counts()[True]
d2 = s_attk.shape[0] - (s_attk == mavg).value_counts()[True]

print("##### diffs #####")
print(f"base-sm: {d1}, best-sm: {d2}")

smoothed = pd.DataFrame(np.array([b_ts,mavg]).T, columns=['time','attack'])
smoothed.to_csv("smoothed.csv", index=False)

def stats(label):
    pos, cnt, lcnt = 0, 0, 0
    for k,g in groupby(label):
        l = len(list(g))
        if k==1:
            if l > 10:
                lcnt += 1
            cnt += 1

        pos += l
    return cnt, lcnt


print(stats(b_attk))
print(stats(s_attk))
print(stats(mavg))

for s in smpath:
    name = str(s)[str(s).find('0.'):]
    sm = pd.read_csv(s, usecols=['attack']).attack
    d = mavg.shape[0] - (mavg == sm).value_counts()[True]
    tapr = etapr.evaluate(anomalies=sm, predictions=mavg)
    print('')
    print(f"+++ {name} +++")
    print(f"diff: {d}")
    print(f"f1: {tapr['f1']:.6f} (tap: {tapr['TaP']:.6f}, tar: {tapr['TaR']:.6f})")

'''
b_win = stats(b_attk)[1]
for w in range(3,12):
    for t in range(1,10):
        t = t/10
        m = b_attk.rolling(w, min_periods=1).mean().apply(lambda s: nanth(s, t))
        m_win = stats(m)[1]
        if m_win >= b_win:
            print(w,t,m_win)
'''
