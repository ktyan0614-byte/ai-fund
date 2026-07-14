# -*- coding: utf-8 -*-
"""掛單價格實驗(動能+營收策略,2020–2026):

週一執行買單的三種掛法:
  A. 開盤市價(基準:一定買到,價格隨市場)
  B. 限價 = 週五收盤 +2%(現行決策卡規則:開盤≤上限照開盤買,盤中回落到上限內用上限買,
     整天都在上限之上就放棄)
  C. 限價 = 週五收盤原價(「掛更低等回落」:只有跌回週五收盤以下才買得到)

假設檢驗:C 省到的價差,是否會被「逆選擇」吃掉——
跌回來讓你買到的,正是走弱的;一路不回頭的,正是最強的。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import config, strategy
from backtest import trade_cost, stats
from data import fetch_fields
from fundamentals import fetch_month_revenue, revenue_yoy_table

ohlc = fetch_fields(list(config.UNIVERSE) + [config.BENCHMARK],
                    start="2020-01-01", cache_prefix="backtest")
close = ohlc["Close"].ffill()
op, lo = ohlc["Open"].ffill(), ohlc["Low"].ffill()
bench = close[config.BENCHMARK]
prices = close[list(config.UNIVERSE)]
rev = fetch_month_revenue(list(config.UNIVERSE), cache_name="month_revenue.csv")
yoy = revenue_yoy_table(rev)


def run(mode, warmup=130):
    dates = close.index
    week_ends = set(pd.Series(dates, index=dates).resample("W-FRI").last().dropna())
    cash, shares = 100_000.0, {}
    pending = None          # (targets, ref_close dict)
    nav_rec, missed, attempts = {}, 0, 0
    for d in dates[warmup:]:
        pxc, pxo, pxl = close.loc[d], op.loc[d], lo.loc[d]
        if pending:
            targets, ref = pending
            pending = None
            # 賣出:一律開盤執行
            for t in list(shares):
                if t not in targets and shares[t] > 0 and not np.isnan(pxo[t]):
                    v = shares[t] * pxo[t]
                    cash += v - trade_cost(v, is_sell=True)
                    shares[t] = 0
            # 買進:依掛單模式
            if targets:
                nav_now = cash + sum(n * pxo[t] for t, n in shares.items() if n > 0)
                per = nav_now / len(targets)
                for t in targets:
                    if np.isnan(pxo[t]):
                        continue
                    cap = {"open": np.inf, "limit2": ref[t] * 1.02,
                           "limit0": ref[t]}[mode]
                    diff = per - shares.get(t, 0) * pxo[t]
                    if diff <= pxo[t]:
                        continue
                    attempts += 1
                    if pxl[t] > cap:          # 全天都在限價之上 → 放棄
                        missed += 1
                        continue
                    fill = min(pxo[t], cap)   # 開盤低於限價就用開盤價
                    buy_n = min(int(diff // fill),
                                int(cash / (fill * (1 + config.FEE_RATE))))
                    if buy_n > 0:
                        c = buy_n * fill
                        shares[t] = shares.get(t, 0) + buy_n
                        cash -= c + trade_cost(c, is_sell=False)
        if d in week_ends:
            targets = strategy.combined_targets(prices, bench, yoy, d)
            targets = [t for t in targets if not np.isnan(pxc.get(t, np.nan))]
            pending = (targets, {t: pxc[t] for t in targets})
        nav_rec[d] = cash + sum(n * pxc[t] for t, n in shares.items()
                                if n > 0 and not np.isnan(pxc[t]))
    return pd.Series(nav_rec), missed, attempts


print("=== 掛單價格對照(動能+營收,2020–2026,含成本)===")
for mode, name in [("open", "A_開盤市價買"), ("limit2", "B_限價收盤+2%(現行)"),
                   ("limit0", "C_限價收盤原價(掛更低)")]:
    nav, missed, attempts = run(mode)
    s = stats(nav)
    print(f"{name:<18} 年化 {s['年化報酬']:>7} | 最大回檔 {s['最大回檔']:>7} | "
          f"夏普 {s['夏普值']} | 買單放棄率 {missed}/{attempts}"
          f"({missed/max(attempts,1):.0%})")
