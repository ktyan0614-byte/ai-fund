# -*- coding: utf-8 -*-
"""動能定義敏感度(2007–2026,動能+營收+濾網):
現行 = 近 60 交易日報酬、不跳月(3-0)
學術標準會跳過最近一個月(短期反轉效應),測試各種 lookback/skip 組合
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


def mom_scores(asof, lookback, skip):
    df = prices.loc[:asof]
    if len(df) < lookback + 1:
        return pd.Series(dtype=float)
    end = df.iloc[-1 - skip] if skip > 0 else df.iloc[-1]
    ret = end / df.iloc[-lookback - 1] - 1
    return ret.dropna().sort_values(ascending=False)


def make_decider(lookback, skip):
    def decide(asof):
        if not strategy.market_ok(bench, asof):
            return []
        mom = mom_scores(asof, lookback, skip)
        mom = mom[mom > 0]
        if mom.empty:
            return []
        f = yoy_asof(yoy, asof).reindex(mom.index).dropna()
        common = mom.index.intersection(f.index)
        if len(common) == 0:
            return list(mom.head(config.TOP_N).index)
        combo = 0.5 * mom[common].rank(pct=True) + 0.5 * f[common].rank(pct=True)
        return list(combo.sort_values(ascending=False).head(config.TOP_N).index)
    return decide


print("=== 動能定義敏感度(動能+營收+濾網,2007–2026)===")
for lb, sk, name in [(60, 0, "3個月不跳月(現行)"), (60, 20, "3個月跳1個月(3-1)"),
                     (120, 20, "6個月跳1個月(6-1)"), (250, 20, "12個月跳1個月(12-1)"),
                     (120, 0, "6個月不跳月")]:
    nav = simulate(prices_all, bench, make_decider(lb, sk))
    s = stats(nav)
    print(f"  {name:<14} 年化 {s['年化報酬']:>7} | 最大回檔 {s['最大回檔']:>7} | 夏普 {s['夏普值']}")
