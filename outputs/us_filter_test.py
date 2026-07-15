# -*- coding: utf-8 -*-
"""美股部位加保命閥測試(2007–2026,台幣計價)

組合:SPY 50%/QQQ 30%/VIG 20%
濾網:SPY(美元價)跌破 N 日均線 → 全部轉現金;站回才進場。每週五檢查。
注意:濾網訊號用美元價判斷(看的是美股趨勢,不讓匯率干擾訊號),
     損益仍以台幣計算。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from backtest import stats

TARGET = {"SPY": 0.5, "QQQ": 0.3, "VIG": 0.2}
COST = 0.001

px = pd.read_csv(os.path.join(os.path.dirname(__file__), "us_etf_twd.csv"),
                 index_col=0, parse_dates=True)

CACHE = os.path.join(os.path.dirname(__file__), "spy_usd.csv")
if os.path.exists(CACHE):
    spy = pd.read_csv(CACHE, index_col=0, parse_dates=True).iloc[:, 0]
else:
    spy = yf.download("SPY", start="2005-06-01", auto_adjust=True,
                      progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy.to_frame("SPY_USD").to_csv(CACHE)
spy = spy.reindex(px.index).ffill()

week_ends = set(pd.Series(px.index, index=px.index).resample("W-FRI").last().dropna())


def run(ma_days):
    cash, units, nav_rec, switches = 100_000.0, {}, {}, 0
    for d in px.index:
        p = px.loc[d]
        nav = cash + sum(u * p[t] for t, u in units.items())
        if d in week_ends:
            s = spy.loc[:d].dropna()
            ok = True if ma_days == 0 or len(s) < ma_days else \
                s.iloc[-1] >= s.iloc[-ma_days:].mean()
            if ok and not units:              # 進場
                for t, w in TARGET.items():
                    units[t] = nav * w * (1 - COST) / p[t]
                cash = 0.0
                switches += 1
            elif not ok and units:            # 出場
                cash = sum(u * p[t] for t, u in units.items()) * (1 - COST)
                units = {}
                switches += 1
            elif units:                       # 在場內:區間再平衡
                weights = {t: units[t] * p[t] / nav for t in TARGET}
                if any(abs(weights[t] - TARGET[t]) > 0.05 for t in TARGET):
                    for t, w in TARGET.items():
                        units[t] = nav * w * (1 - COST) / p[t]
                    cash = 0.0
        nav_rec[d] = cash + sum(u * p[t] for t, u in units.items())
    return pd.Series(nav_rec), switches


print("=== 美股組合加保命閥(SPY 美元價 vs 均線,2007–2026,台幣)===")
for ma, name in [(0, "無濾網(現行帳戶四)"), (120, "120日均線濾網"), (200, "200日均線濾網")]:
    nav, sw = run(ma)
    s = stats(nav)
    print(f"  {name:<12} 年化 {s['年化報酬']:>7} | 最大回檔 {s['最大回檔']:>7} | "
          f"夏普 {s['夏普值']} | 進出 {sw} 次")
    yearly = nav.resample("YE").last().pct_change(fill_method=None)
    yearly.index = yearly.index.year
    if ma:
        print(f"    關鍵年: 2008 {yearly.loc[2008]:+.0%} | 2020 {yearly.loc[2020]:+.0%} | "
              f"2022 {yearly.loc[2022]:+.0%} | 2024 {yearly.loc[2024]:+.0%}")
    else:
        print(f"    關鍵年: 2008 {yearly.loc[2008]:+.0%} | 2020 {yearly.loc[2020]:+.0%} | "
              f"2022 {yearly.loc[2022]:+.0%} | 2024 {yearly.loc[2024]:+.0%}")
