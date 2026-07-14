# -*- coding: utf-8 -*-
"""動能+營收策略(帳戶二):19 年回測的完整持股軌跡
輸出:每年主要持股、進出統計、關鍵時期(2008/2021/2022)的實際動作
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import config, strategy
from data import fetch_prices
from fundamentals import fetch_month_revenue, revenue_yoy_table

TWII = "^TWII"
prices_all = fetch_prices(list(config.UNIVERSE) + [TWII], start="2006-06-01",
                          cache_name="long_prices.csv").ffill()
bench = prices_all[TWII]
prices = prices_all[list(config.UNIVERSE)]
rev = fetch_month_revenue(list(config.UNIVERSE), start="2005-06-01",
                          cache_name="month_revenue_long.csv")
yoy_table = revenue_yoy_table(rev)

dates = prices_all.index
week_ends = pd.Series(dates, index=dates).resample("W-FRI").last().dropna()
week_ends = [d for d in week_ends if dates.get_loc(d) >= 130]

records = []
prev = set()
for d in week_ends:
    targets = set(strategy.combined_targets(prices, bench, yoy_table, d))
    records.append({"date": d, "targets": targets,
                    "buys": targets - prev, "sells": prev - targets,
                    "cash": len(targets) == 0})
    prev = targets

df = pd.DataFrame(records).set_index("date")


def zh(t):
    return config.UNIVERSE[t][0]


n = len(df)
cash_weeks = df["cash"].sum()
total_buys = sum(len(b) for b in df["buys"])
print(f"回測共 {n} 週({df.index[0].date()} ~ {df.index[-1].date()})")
print(f"空手(濾網觸發)週數:{cash_weeks}({cash_weeks/n:.0%})")
print(f"累計買進動作:{total_buys} 次,平均每年換股 {total_buys/19:.0f} 次")

# 每檔股票被持有的總週數
weeks_held = {}
for targets in df["targets"]:
    for t in targets:
        weeks_held[t] = weeks_held.get(t, 0) + 1
top = sorted(weeks_held.items(), key=lambda x: -x[1])
print("\n=== 19 年來被持有最久的股票 ===")
for t, w in top[:12]:
    print(f"  {zh(t)}({t.replace('.TW','')}): {w} 週(約 {w/52:.1f} 年,"
          f"占投資期間 {w/n:.0%})")

# 每年最常持有的前三檔
print("\n=== 各年度最常持有(前三)===")
for year, g in df.groupby(df.index.year):
    cnt = {}
    for targets in g["targets"]:
        for t in targets:
            cnt[t] = cnt.get(t, 0) + 1
    top3 = sorted(cnt.items(), key=lambda x: -x[1])[:3]
    cash_pct = g["cash"].mean()
    names = "、".join(f"{zh(t)}({w}週)" for t, w in top3) if top3 else "全年空手"
    print(f"  {year}: {names}" + (f" | 空手{cash_pct:.0%}" if cash_pct > 0.2 else ""))

# 關鍵時期
print("\n=== 2008 金融海嘯:濾網動作 ===")
y08 = df.loc["2007-10":"2009-08"]
state = None
for d, row in y08.iterrows():
    s = "空手" if row["cash"] else "持股"
    if s != state:
        print(f"  {d.date()} → {s}" +
              ("" if row["cash"] else f":{'、'.join(zh(t) for t in row['targets'])}"))
        state = s

print("\n=== 2021 航運年:長榮/陽明的進出 ===")
for d, row in df.loc["2020-06":"2022-06"].iterrows():
    ships = {t for t in row["buys"] | row["sells"] if t in ("2603.TW", "2609.TW")}
    for t in ships:
        act = "買進" if t in row["buys"] else "賣出"
        px = prices.loc[:d, t].iloc[-1]
        print(f"  {d.date()} {act} {zh(t)} @ {px:.1f}")
