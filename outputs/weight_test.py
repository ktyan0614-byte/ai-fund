# -*- coding: utf-8 -*-
"""兩個檢驗(動能+營收,2007–2026):

1. 權重敏感度:動能 30%~100% 掃一遍,看 50/50 是不是碰巧好看的魔法數字
2. 營收的「實際影響力」:混合策略每週的前五名,跟純動能的前五名平均重疊幾檔?
   (權重寫 50% 不代表決策被改變 50%)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import config, strategy
from backtest import simulate, stats
from data import fetch_prices
from fundamentals import fetch_month_revenue, revenue_yoy_table, yoy_asof

TWII = "^TWII"
prices_all = fetch_prices(list(config.UNIVERSE) + [TWII], start="2006-06-01",
                          cache_name="long_prices.csv").ffill()
bench = prices_all[TWII]
prices = prices_all[list(config.UNIVERSE)]
rev = fetch_month_revenue(list(config.UNIVERSE), start="2005-06-01",
                          cache_name="month_revenue_long.csv")
yoy = revenue_yoy_table(rev)


def blend_targets(asof, w_mom):
    if not strategy.market_ok(bench, asof):
        return []
    mom = strategy.momentum_scores(prices, asof)
    mom = mom[mom > 0]
    if mom.empty:
        return []
    f = yoy_asof(yoy, asof).reindex(mom.index).dropna()
    common = mom.index.intersection(f.index)
    if len(common) == 0:
        return list(mom.head(config.TOP_N).index)
    combo = w_mom * mom[common].rank(pct=True) + (1 - w_mom) * f[common].rank(pct=True)
    return list(combo.sort_values(ascending=False).head(config.TOP_N).index)


print("=== 1. 權重敏感度(動能權重 30%→100%)===")
for w in [0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
    nav = simulate(prices_all, bench, lambda asof, w=w: blend_targets(asof, w))
    s = stats(nav)
    print(f"  動能 {w:.0%} / 營收 {1-w:.0%}: 年化 {s['年化報酬']:>7} | "
          f"最大回檔 {s['最大回檔']:>7} | 夏普 {s['夏普值']}")

print("\n=== 2. 營收的實際影響力 ===")
dates = prices_all.index
week_ends = pd.Series(dates, index=dates).resample("W-FRI").last().dropna()
week_ends = [d for d in week_ends if dates.get_loc(d) >= 130]
overlaps, both_hold = [], 0
for d in week_ends:
    c = set(blend_targets(d, 1.0))    # 純動能
    e = set(blend_targets(d, 0.5))    # 混合
    if c and e:
        overlaps.append(len(c & e))
        both_hold += 1
avg = sum(overlaps) / len(overlaps)
print(f"有持股的 {both_hold} 週中,混合策略前五名與純動能前五名平均重疊 {avg:.2f} 檔")
print(f"→ 營收因子平均每週只改寫 {5-avg:.2f} 檔({(5-avg)/5:.0%})的決策")
