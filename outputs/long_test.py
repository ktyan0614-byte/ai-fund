# -*- coding: utf-8 -*-
"""長期回測(2007–2026,約 19 年):涵蓋 2008 金融海嘯、2011 歐債、
2015、2018、2020 疫情崩盤、2022 升息空頭——回答「6 年多頭資料夠不夠」。

注意兩個誠實的限制:
1. 大盤改用加權指數(^TWII,不含股息),持有指數的報酬被低估約每年 3-4%
2. 倖存者偏誤:投資範圍是「2026 年的權值股」,2007 年的你不可能未卜先知選中它們
   → 長期絕對報酬被高估,這個測試的重點是「策略邏輯在空頭年會不會壞掉」,
     不是預期報酬有多少
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import config, strategy
from backtest import simulate, stats
from data import fetch_prices
from fundamentals import fetch_month_revenue, revenue_yoy_table, yoy_asof

TWII = "^TWII"
tickers = list(config.UNIVERSE) + [TWII]
print("下載 2006 起的長期股價...")
prices_all = fetch_prices(tickers, start="2006-06-01",
                          cache_name="long_prices.csv").ffill()
bench = prices_all[TWII]
prices = prices_all[[c for c in prices_all.columns if c != TWII]]
print(f"資料區間: {prices_all.index[0].date()} ~ {prices_all.index[-1].date()}, "
      f"{len(prices_all)} 個交易日")

print("下載 2005 起的月營收...")
rev = fetch_month_revenue(list(config.UNIVERSE), start="2005-06-01",
                          cache_name="month_revenue_long.csv")
yoy_table = revenue_yoy_table(rev)


def decide_A(asof):   # 持有大盤指數(不含息)
    return [TWII]


def decide_B(asof):   # 純動能,無濾網
    s = strategy.momentum_scores(prices, asof)
    s = s[s > 0]
    return list(s.head(config.TOP_N).index)


def decide_C(asof):   # 動能 + 大盤濾網
    return strategy.target_holdings(prices, bench, asof)


def decide_E(asof):   # 動能+營收混合 + 濾網
    return strategy.combined_targets(prices, bench, yoy_table, asof)


navs = {}
for name, fn in [("A_持有大盤(不含息)", decide_A), ("B_動能無濾網", decide_B),
                 ("C_動能+濾網", decide_C), ("E_動能+營收+濾網", decide_E)]:
    print(f"回測 {name} ...")
    navs[name] = simulate(prices_all, bench, fn)

print(f"\n=== 長期回測 2007–2026(初始 10 萬,含交易成本)===")
for name, nav in navs.items():
    s = stats(nav)
    print(f"{name:<14} 年化 {s['年化報酬']:>7} | 最大回檔 {s['最大回檔']:>7} | "
          f"夏普 {s['夏普值']} | 期末 {nav.iloc[-1]:>12,.0f}")

result = pd.DataFrame(navs)
yearly = result.resample("YE").last().pct_change(fill_method=None)
yearly.iloc[0] = result.resample("YE").last().iloc[0] / result.dropna().iloc[0] - 1
yearly.index = yearly.index.year
print("\n=== 各年度報酬(重點看空頭年)===")
print(yearly.map(lambda x: f"{x:+.0%}" if pd.notna(x) else "").to_string())
result.to_csv(os.path.join(os.path.dirname(__file__), "long_test_navs.csv"),
              encoding="utf-8-sig")
