# -*- coding: utf-8 -*-
"""因子實驗:毛利率趨勢、獲利門檻,對照現有兩策略(2007–2026)

候選:
  G1 動能+毛利趨勢 50/50    (毛利因子單獨上場,看它自己行不行)
  G2 動能1/2+營收1/4+毛利1/4 (三因子混合)
  H  帳戶二策略 + 近四季獲利>0 門檻
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import config, strategy
from backtest import simulate, stats
from data import fetch_prices
from fundamentals import (fetch_month_revenue, revenue_yoy_table, yoy_asof,
                          fetch_financials, margin_trend_table, profitable_table)

TWII = "^TWII"
tickers = list(config.UNIVERSE) + [TWII]
prices_all = fetch_prices(tickers, start="2006-06-01",
                          cache_name="long_prices.csv").ffill()
bench = prices_all[TWII]
prices = prices_all[[c for c in prices_all.columns if c != TWII]]

rev = fetch_month_revenue(list(config.UNIVERSE), start="2005-06-01",
                          cache_name="month_revenue_long.csv")
yoy_table = revenue_yoy_table(rev)

print("下載季報資料...")
fin = fetch_financials(list(config.UNIVERSE), start="2005-01-01",
                       cache_name="financials_long.csv")
gm_table = margin_trend_table(fin)
ok_table = profitable_table(fin)
print(f"季報資料: {fin['qend'].min().date()} ~ {fin['qend'].max().date()},"
      f" {fin['ticker'].nunique()} 檔")


def blend(asof, weights):
    """通用多因子混合:weights = {'mom': w, 'rev': w, 'gm': w}"""
    mom = strategy.momentum_scores(prices, asof)
    mom = mom[mom > 0]
    if mom.empty:
        return pd.Series(dtype=float)
    parts = {"mom": mom.rank(pct=True)}
    if weights.get("rev"):
        f = yoy_asof(yoy_table, asof).reindex(mom.index).dropna()
        parts["rev"] = f.rank(pct=True)
    if weights.get("gm"):
        f = yoy_asof(gm_table, asof).reindex(mom.index).dropna()
        parts["gm"] = f.rank(pct=True)
    common = parts["mom"].index
    for s in parts.values():
        common = common.intersection(s.index)
    if len(common) == 0:
        return mom
    total = sum(weights[k] * parts[k][common] for k in parts)
    return total.sort_values(ascending=False)


def make_decider(weights, profit_gate=False):
    def decide(asof):
        if not strategy.market_ok(bench, asof):
            return []
        combo = blend(asof, weights)
        if profit_gate:
            ttm = yoy_asof(ok_table, asof)
            # 只剔除「確定近四季虧損」的;資料缺漏(NaN)視為通過
            combo = combo[[t for t in combo.index
                           if not (pd.notna(ttm.get(t)) and ttm.get(t) <= 0)]]
        return list(combo.head(config.TOP_N).index)
    return decide


runs = {
    "C_純動能":        make_decider({"mom": 1.0}),
    "E_動能+營收":      make_decider({"mom": .5, "rev": .5}),
    "G1_動能+毛利":     make_decider({"mom": .5, "gm": .5}),
    "G2_三因子混合":     make_decider({"mom": .5, "rev": .25, "gm": .25}),
    "H_E+獲利門檻":     make_decider({"mom": .5, "rev": .5}, profit_gate=True),
}

navs = {}
for name, fn in runs.items():
    print(f"回測 {name} ...")
    navs[name] = simulate(prices_all, bench, fn)

print("\n=== 因子實驗結果(2007–2026,含交易成本)===")
for name, nav in navs.items():
    s = stats(nav)
    print(f"{name:<10} 年化 {s['年化報酬']:>7} | 最大回檔 {s['最大回檔']:>7} | "
          f"夏普 {s['夏普值']} | 期末 {nav.iloc[-1]:>12,.0f}")

result = pd.DataFrame(navs)
yearly = result.resample("YE").last().pct_change(fill_method=None)
yearly.iloc[0] = result.resample("YE").last().iloc[0] / result.dropna().iloc[0] - 1
yearly.index = yearly.index.year
print("\n=== 各年度報酬 ===")
print(yearly.map(lambda x: f"{x:+.0%}" if pd.notna(x) else "").to_string())
