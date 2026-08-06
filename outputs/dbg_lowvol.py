# -*- coding: utf-8 -*-
"""除錯:低波動組為何從 2013 起淨值不動"""
import json, os, sys
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEJ = os.path.join(BASE, "tej"); sys.path.insert(0, BASE)

big = pd.read_csv(os.path.join(TEJ, "prices_daily_merged.csv"), encoding="utf-8-sig",
                  parse_dates=["date"], dtype={"code": str})
mid = pd.read_csv(os.path.join(TEJ, "prices_daily_midcap_merged.csv"), encoding="utf-8-sig",
                  parse_dates=["date"], dtype={"code": str})
allp = pd.concat([big, mid]).drop_duplicates(["code", "date"])
wide = allp.pivot(index="date", columns="code", values="close").sort_index()
last_valid = wide.apply(lambda s: s.last_valid_index())
px = wide.ffill(limit=15)
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

for ds in ["2010-12-31", "2013-06-28", "2015-12-31", "2020-12-31"]:
    d = pd.Timestamp(ds)
    uni = universe_at(d)
    win = px.loc[:d, uni].tail(251)
    vol = win.pct_change(fill_method=None).std().dropna()
    picks = list(vol.nsmallest(10).index)
    print(f"{ds} | 池 {len(uni):>3} | 有波動值 {len(vol):>3}")
    for c in picks[:6]:
        n_obs = win[c].notna().sum()
        print(f"    {names.get(c,c):<10} vol={vol[c]:.6f}  有效報價天數={n_obs}  "
              f"最新價={px.loc[:d, c].iloc[-1]:.2f}")
