# -*- coding: utf-8 -*-
"""指數+濾網測試:持有大盤(含息),跌破均線轉現金——擇時的獨立審判

用 TEJ 加權報酬指數(Y9997,含息)2006–2026。
以 0050 實際費率近似(手續費 0.0855%+ETF 證交稅 0.1%,單邊約 0.15%)。
每週五收盤檢查(與系統一致)。
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
full = pd.read_csv(os.path.join(BASE, "tej", "prices_daily_merged.csv"),
                   encoding="utf-8-sig", parse_dates=["date"], dtype={"code": str})
bench = full[full["code"] == "Y9997"].set_index("date")["close"].sort_index()
dates = bench.index
COST = 0.0015


def run(ma):
    week_ends = set(pd.Series(dates, index=dates).resample("W-FRI").last().dropna())
    nav, in_mkt, units, cash = {}, False, 0.0, 100_000.0
    switches = 0
    for d in dates[210:]:
        p = bench.loc[d]
        if d in week_ends:
            s = bench.loc[:d]
            ok = ma == 0 or s.iloc[-1] >= s.iloc[-ma:].mean()
            if ok and not in_mkt:
                units = cash * (1 - COST) / p
                cash, in_mkt = 0.0, True
                switches += 1
            elif not ok and in_mkt:
                cash = units * p * (1 - COST)
                units, in_mkt = 0.0, False
                switches += 1
        nav[d] = cash + units * p
    return pd.Series(nav), switches


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    r = nav.pct_change(fill_method=None).dropna()
    return cagr, dd, r.mean() / r.std() * np.sqrt(252)


navs = {}
print("=== 大盤含息 ± 均線濾網(2007–2026,週頻檢查,含 0050 級費率)===")
for ma, name in [(0, "買進持有(基準)"), (120, "120日均線濾網"), (200, "200日均線濾網")]:
    nav, sw = run(ma)
    navs[name] = nav
    c, d, sh = stats(nav)
    print(f"{name:<12} 年化 {c:+.1%} | 最大回檔 {d:.1%} | 夏普 {sh:.2f} | "
          f"進出 {sw} 次")

res = pd.DataFrame(navs).dropna()
yearly = res.resample("YE").last().pct_change(fill_method=None)
yearly.iloc[0] = res.resample("YE").last().iloc[0] / res.iloc[0] - 1
yearly.index = yearly.index.year
key = yearly.loc[yearly.index.isin([2008, 2009, 2011, 2015, 2018, 2020, 2022, 2024])]
print("\n關鍵年度:")
print(key.map(lambda x: f"{x:+.0%}" if pd.notna(x) else "").to_string())

print("\n滾動五年視窗:")
monthly = res.resample("ME").last()
for name in navs:
    m = monthly[name]
    r5 = ((m.shift(-60) / m) ** (1 / 5) - 1).dropna()
    print(f"{name:<12} 最差 {r5.min():+.1%} | 中位 {r5.median():+.1%} | "
          f"最佳 {r5.max():+.1%} | 五年虧損率 {(r5 < 0).mean():.0%}")
