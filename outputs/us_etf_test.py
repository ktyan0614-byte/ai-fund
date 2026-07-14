# -*- coding: utf-8 -*-
"""美股 ETF 部門回測(台幣計價,2007–2026)

配置:SPY 50% / QQQ 30% / VIG 20%,被動持有
再平衡:每週檢查,任一 ETF 偏離目標配置 >5 個百分點才調回(band rebalancing)
成本:每筆 0.1%(海外券商費率,偏保守)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from backtest import stats

TARGET = {"SPY": 0.5, "QQQ": 0.3, "VIG": 0.2}
BAND, COST = 0.05, 0.001

CACHE = os.path.join(os.path.dirname(__file__), "us_etf_twd.csv")
if os.path.exists(CACHE):
    px = pd.read_csv(CACHE, index_col=0, parse_dates=True)
else:
    raw = yf.download(list(TARGET) + ["TWD=X"], start="2006-06-01",
                      auto_adjust=True, progress=False)["Close"]
    fx = raw["TWD=X"]
    fx[fx.pct_change().abs() > 0.15] = float("nan")
    fx = fx.ffill()
    px = raw[list(TARGET)].mul(fx, axis=0).dropna().ffill()
    px.to_csv(CACHE)

week_ends = set(pd.Series(px.index, index=px.index).resample("W-FRI").last().dropna())
cash, units, nav_rec, rebalances = 100_000.0, {}, {}, 0

for d in px.index:
    p = px.loc[d]
    nav = cash + sum(u * p[t] for t, u in units.items())
    if d in week_ends:
        if not units:                      # 建倉
            for t, w in TARGET.items():
                units[t] = nav * w * (1 - COST) / p[t]
            cash = 0.0
        else:
            weights = {t: units[t] * p[t] / nav for t in TARGET}
            if any(abs(weights[t] - TARGET[t]) > BAND for t in TARGET):
                for t, w in TARGET.items():
                    units[t] = nav * w * (1 - COST) / p[t]
                cash = 0.0
                rebalances += 1
    nav_rec[d] = cash + sum(u * p[t] for t, u in units.items())

nav_s = pd.Series(nav_rec)
s = stats(nav_s)
print("=== 美股 ETF 部門(SPY 50/QQQ 30/VIG 20,台幣計價)===")
print(f"年化 {s['年化報酬']} | 最大回檔 {s['最大回檔']} | 夏普 {s['夏普值']} | "
      f"期末 {nav_s.iloc[-1]:,.0f} | 19 年再平衡 {rebalances} 次")
yearly = nav_s.resample("YE").last().pct_change(fill_method=None)
yearly.index = yearly.index.year
print(yearly.dropna().map(lambda x: f"{x:+.0%}").to_string())
