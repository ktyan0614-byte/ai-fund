# -*- coding: utf-8 -*-
"""Antonacci 雙動能資產輪動(GEM)回測,台幣視角(2007–2026)

規則(每月最後交易日檢查一次):
  1. 算美國股市(SPY)與國際股市(EFA)過去 12 個月報酬
  2. 兩者中較強者若 12 個月報酬 > 0 → 持有它
  3. 兩者 12 個月報酬皆 ≤ 0 → 轉進美國綜合債券(AGG)
全程只持有一檔 ETF。價格以美元價 × 美元兌台幣匯率換算(含匯率損益)。
成本:每次轉換以 0.3% 計(手續費+價差,偏保守)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from backtest import stats

CACHE = os.path.join(os.path.dirname(__file__), "gem_twd.csv")
if os.path.exists(CACHE):
    px = pd.read_csv(CACHE, index_col=0, parse_dates=True)
else:
    raw = yf.download(["SPY", "EFA", "AGG", "TWD=X"], start="2005-06-01",
                      auto_adjust=True, progress=False)["Close"]
    fx = raw["TWD=X"]
    r = fx.pct_change()
    fx[r.abs() > 0.15] = float("nan")     # 清匯率壞點
    fx = fx.ffill()
    px = raw[["SPY", "EFA", "AGG"]].mul(fx, axis=0).dropna(how="all").ffill()
    px.to_csv(CACHE)

month_ends = px.resample("ME").last().index
month_ends = [px.index[px.index.get_indexer([m], method="ffill")[0]] for m in month_ends]

COST = 0.003
cash_units = None      # 以「持有單位數」記帳
hold, nav, nav_rec, switches = None, 100_000.0, {}, 0

for d in px.index:
    if hold is not None:
        pass
    if d in month_ends:
        win = px.loc[:d]
        if len(win) > 260:
            r12 = {t: win[t].iloc[-1] / win[t].iloc[-253] - 1 for t in ["SPY", "EFA"]}
            best = max(r12, key=r12.get)
            target = best if r12[best] > 0 else "AGG"
            if target != hold:
                if hold is not None:
                    nav = units * px.loc[d, hold] * (1 - COST)
                units = nav * (1 - COST) / px.loc[d, target]
                hold = target
                switches += 1
    if hold is not None:
        nav_rec[d] = units * px.loc[d, hold]

nav_s = pd.Series(nav_rec)
s = stats(nav_s)
print("=== GEM 全球雙動能(台幣計價,2007–2026)===")
print(f"年化 {s['年化報酬']} | 最大回檔 {s['最大回檔']} | 夏普 {s['夏普值']} | "
      f"期末 {nav_s.iloc[-1]:,.0f} | 19 年共切換 {switches} 次")

yearly = nav_s.resample("YE").last().pct_change(fill_method=None)
yearly.index = yearly.index.year
hold_hist = pd.Series({d: h for d, h in nav_rec.items()})
print("\n各年度報酬:")
print(yearly.dropna().map(lambda x: f"{x:+.0%}").to_string())
