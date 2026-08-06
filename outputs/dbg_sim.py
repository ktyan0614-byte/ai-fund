# -*- coding: utf-8 -*-
"""除錯:低波動組模擬迴圈,逐季印出現金/持股/淨值"""
import json, os, sys
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEJ = os.path.join(BASE, "tej"); sys.path.insert(0, BASE)
FEE, TAX, SLIP = 0.001425 * 0.6, 0.003, 0.003

big = pd.read_csv(os.path.join(TEJ, "prices_daily_merged.csv"), encoding="utf-8-sig",
                  parse_dates=["date"], dtype={"code": str})
mid = pd.read_csv(os.path.join(TEJ, "prices_daily_midcap_merged.csv"), encoding="utf-8-sig",
                  parse_dates=["date"], dtype={"code": str})
allp = pd.concat([big, mid]).drop_duplicates(["code", "date"])
wide = allp.pivot(index="date", columns="code", values="close").sort_index()
last_valid = wide.apply(lambda s: s.last_valid_index())
px = wide.ffill(limit=15)
dates = px.index
u1 = json.load(open(os.path.join(TEJ, "universe_history.json"), encoding="utf-8"))
u2 = json.load(open(os.path.join(TEJ, "universe_midcap.json"), encoding="utf-8"))
q_keys = sorted(set(u1["quarters"]) & set(u2["quarters"]))
quarters = {q: list(dict.fromkeys(u1["quarters"][q] + u2["quarters"][q])) for q in q_keys}
q_starts = pd.to_datetime(q_keys, format="%Y-%m")
names = {**u1["names"], **u2["names"]}


def universe_at(d):
    i = q_starts.searchsorted(d, side="right") - 1
    if i < 0: return []
    return [c for c in quarters[q_keys[i]] if c in px.columns
            and last_valid[c] is not None and last_valid[c] >= d - pd.Timedelta(days=5)]


def pick_lowvol(d, n=10):
    uni = universe_at(d)
    win = px.loc[:d, uni].tail(251)
    if len(win) < 200: return []
    vol = win.pct_change(fill_method=None).std().dropna()
    return list(vol.nsmallest(n).index)


q_ends = pd.Series(dates, index=dates).resample("QE").last().dropna()
q_ends = [d for d in q_ends if dates.get_loc(d) >= 260]
cash, shares = 100_000.0, {}
reb = set(q_ends)
for d in dates[dates.get_loc(q_ends[0]):]:
    p = px.loc[d]
    if d in reb:
        tg = [t for t in pick_lowvol(d) if not np.isnan(p.get(t, np.nan))]
        for t in list(shares):
            if t not in tg and shares[t] > 0 and not np.isnan(p[t]):
                v = shares[t] * p[t]
                cash += v - (max(1, v * FEE) + v * TAX + v * SLIP)
                shares[t] = 0
        if tg:
            held_val = sum(n * p[t] for t, n in shares.items() if n > 0)
            nav = cash + held_val
            w = 1 / len(tg)
            for t in tg:
                diff = nav * w - shares.get(t, 0) * p[t]
                if diff > p[t]:
                    n = min(int(diff // p[t]), int(cash / (p[t] * (1 + FEE + SLIP))))
                    if n > 0:
                        v = n * p[t]
                        shares[t] = shares.get(t, 0) + n
                        cash -= v + max(1, v * FEE) + v * SLIP
                elif diff < -p[t] and shares.get(t, 0) > 0:
                    n = min(int(-diff // p[t]), shares[t])
                    if n > 0:
                        v = n * p[t]
                        cash += v - (max(1, v * FEE) + v * TAX + v * SLIP)
                        shares[t] -= n
            if d.year in (2011, 2012, 2013, 2014) and d.month in (6, 12):
                nz = {t: n for t, n in shares.items() if n > 0}
                nanp = [t for t in nz if np.isnan(p[t])]
                print(f"{d.date()} 現金 {cash:>10,.0f} | 持股 {len(nz)} 檔 | "
                      f"持股市值 {sum(n*p[t] for t,n in nz.items() if not np.isnan(p[t])):>10,.0f} | "
                      f"NaN報價持股 {len(nanp)} | 目標 {len(tg)} 檔")
                if nanp:
                    print("      NaN:", [names.get(t, t) for t in nanp])
