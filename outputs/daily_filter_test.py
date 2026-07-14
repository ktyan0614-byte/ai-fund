# -*- coding: utf-8 -*-
"""測試:大盤濾網「每日檢查、跌破當天就賣」vs「每週五才檢查」(2007–2026)

假設:週間跌破均線立刻出場能躲更多下跌?
代價:均線附近的震盪會造成反覆進出(whipsaw),多付成本、錯過反彈。
讓數據說話。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import config, strategy
from backtest import simulate, stats, trade_cost
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


def decide_E(asof):
    return strategy.combined_targets(prices, bench, yoy_table, asof)


def simulate_daily_filter(initial_cash=100_000, warmup=130):
    """每週五換股,但濾網每天收盤檢查:跌破當天全賣,回穩後下個週五才重新進場。"""
    dates = prices_all.index
    week_ends = set(pd.Series(dates, index=dates).resample("W-FRI").last().dropna())
    start_i = warmup
    cash, shares = float(initial_cash), {}
    nav_records = {}
    exits = 0
    for d in dates[start_i:]:
        px = prices_all.loc[d]
        ok = strategy.market_ok(bench, d)
        if not ok and any(n > 0 for n in shares.values()):   # 當日跌破 → 立刻清倉
            for t in list(shares):
                if shares[t] > 0 and not np.isnan(px[t]):
                    v = shares[t] * px[t]
                    cash += v - trade_cost(v, is_sell=True)
                    shares[t] = 0
            exits += 1
        if d in week_ends and ok:                            # 週五且濾網通過 → 換股
            targets = [t for t in decide_E(d) if not np.isnan(px.get(t, np.nan))]
            for t in list(shares):
                if t not in targets and shares[t] > 0 and not np.isnan(px[t]):
                    v = shares[t] * px[t]
                    cash += v - trade_cost(v, is_sell=True)
                    shares[t] = 0
            if targets:
                nav_now = cash + sum(n * px[t] for t, n in shares.items() if n > 0)
                per = nav_now / len(targets)
                for t in targets:
                    cur = shares.get(t, 0) * px[t]
                    diff = per - cur
                    if diff > px[t]:
                        buy_n = min(int(diff // px[t]),
                                    int(cash / (px[t] * (1 + config.FEE_RATE))))
                        if buy_n > 0:
                            c = buy_n * px[t]
                            shares[t] = shares.get(t, 0) + buy_n
                            cash -= c + trade_cost(c, is_sell=False)
        nav_records[d] = cash + sum(n * px[t] for t, n in shares.items()
                                    if n > 0 and not np.isnan(px[t]))
    return pd.Series(nav_records), exits


nav_weekly = simulate(prices_all, bench, decide_E)
nav_daily, exits = simulate_daily_filter()

print("=== 濾網檢查頻率對照(動能+營收,2007–2026)===")
for name, nav in [("每週五檢查(現行)", nav_weekly), ("每日檢查,跌破即賣", nav_daily)]:
    s = stats(nav)
    print(f"{name:<14} 年化 {s['年化報酬']:>7} | 最大回檔 {s['最大回檔']:>7} | 夏普 {s['夏普值']}")
print(f"\n每日版 19 年共觸發盤中清倉 {exits} 次(平均每年 {exits/19:.1f} 次)")
