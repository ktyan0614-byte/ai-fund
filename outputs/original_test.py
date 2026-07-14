# -*- coding: utf-8 -*-
"""原創策略實驗(2007–2026):三個從市場結構推導的假設,對照現有策略

  O1 營收加速度:買「成長率正在加速」的股票(二階導數假設)
  O2 潛伏者    :營收在加速、但股價還沒漲的股票(注意力落差假設)
  O3 績優股回彈 :有獲利的公司裡,買從半年高點跌最深的(Eric 的直覺,系統化版)

對照組:純動能(帳戶一)、動能+毛利(帳戶三)、大盤
所有策略同樣週頻、同樣費稅、同樣大盤濾網。
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
fin = fetch_financials(list(config.UNIVERSE), start="2005-01-01",
                       cache_name="financials_long.csv")
gm_table = margin_trend_table(fin)
ttm_table = profitable_table(fin)


def accel_asof(asof, gap_days=91):
    """營收年增率的「加速度」:目前的年增率 − 約一季前的年增率。"""
    now = yoy_asof(yoy_table, asof)
    before = yoy_asof(yoy_table, pd.Timestamp(asof) - pd.Timedelta(days=gap_days))
    return (now - before).dropna()


def quality_pass(asof, tickers_):
    """獲利門檻:近四季合計虧損的剔除(資料缺漏視為通過)。"""
    ttm = yoy_asof(ttm_table, asof)
    return [t for t in tickers_
            if not (pd.notna(ttm.get(t)) and ttm.get(t) <= 0)]


def decide_O1(asof):   # 營收加速度
    if not strategy.market_ok(bench, asof):
        return []
    acc = accel_asof(asof)
    acc = acc[acc > 0]
    return list(acc.sort_values(ascending=False).head(config.TOP_N).index)


def decide_O2(asof):   # 潛伏者:加速中但股價落後
    if not strategy.market_ok(bench, asof):
        return []
    acc = accel_asof(asof)
    acc = acc[acc > 0]
    mom = strategy.momentum_scores(prices, asof)
    cands = acc.index.intersection(mom.index)
    if len(cands) == 0:
        return []
    return list(mom[cands].sort_values().head(config.TOP_N).index)  # 挑最沒漲的


def decide_O3(asof):   # 績優股回彈:獲利公司中,距 120 日高點最深者
    if not strategy.market_ok(bench, asof):
        return []
    window = prices.loc[:asof].iloc[-120:]
    dd = (window.iloc[-1] / window.max() - 1).dropna()
    dd = dd[quality_pass(asof, dd.index)]
    return list(dd.sort_values().head(config.TOP_N).index)  # 跌最深的 5 檔


def decide_C(asof):    # 對照:純動能
    return strategy.target_holdings(prices, bench, asof)


def decide_G1(asof):   # 對照:動能+毛利(帳戶三)
    return strategy.combined_targets(prices, bench, gm_table, asof)


def decide_MKT(asof):  # 對照:大盤
    return [TWII]


runs = {
    "O1_營收加速度": decide_O1,
    "O2_潛伏者":   decide_O2,
    "O3_績優股回彈": decide_O3,
    "對照_純動能":   decide_C,
    "對照_動能+毛利": decide_G1,
    "對照_大盤不含息": decide_MKT,
}

navs = {}
for name, fn in runs.items():
    print(f"回測 {name} ...")
    navs[name] = simulate(prices_all, bench, fn)

print("\n=== 原創策略實驗(2007–2026,含交易成本)===")
for name, nav in navs.items():
    s = stats(nav)
    print(f"{name:<11} 年化 {s['年化報酬']:>7} | 最大回檔 {s['最大回檔']:>7} | "
          f"夏普 {s['夏普值']} | 期末 {nav.iloc[-1]:>12,.0f}")

result = pd.DataFrame(navs)
yearly = result.resample("YE").last().pct_change(fill_method=None)
yearly.iloc[0] = result.resample("YE").last().iloc[0] / result.dropna().iloc[0] - 1
yearly.index = yearly.index.year
print("\n=== 各年度報酬 ===")
print(yearly.map(lambda x: f"{x:+.0%}" if pd.notna(x) else "").to_string())
