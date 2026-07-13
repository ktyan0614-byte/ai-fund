# -*- coding: utf-8 -*-
"""兩個實驗:
1. 週轉率統計:動能策略每週實際換幾檔股票?平均持有多久?
2. 濾網轉債券版:濾網觸發時,抱現金 vs 轉進美國長天期公債(TLT,換算台幣)

TLT 是美股 ETF,這裡把它的美元價格乘上美元兌台幣匯率,模擬台灣投資人的實際報酬
(國內類似商品 00679B 到 2017 年才上市,歷史太短,用 TLT 代理)。
交易成本沿用台股股票費率,略偏保守。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
import config, strategy
from backtest import simulate, stats
from data import fetch_prices
from fundamentals import fetch_month_revenue, revenue_yoy_table

TWII = "^TWII"
BOND = "TLT_TWD"

tickers = list(config.UNIVERSE) + [TWII]
prices_all = fetch_prices(tickers, start="2006-06-01",
                          cache_name="long_prices.csv").ffill()
bench = prices_all[TWII]
prices = prices_all[[c for c in prices_all.columns if c != TWII]]

# TLT(美元)換算台幣
cache = os.path.join(os.path.dirname(__file__), "tlt_twd.csv")
if os.path.exists(cache):
    tlt_twd = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
else:
    raw = yf.download(["TLT", "TWD=X"], start="2006-06-01",
                      auto_adjust=True, progress=False)["Close"]
    tlt_twd = (raw["TLT"] * raw["TWD=X"]).dropna()
    # Yahoo 匯率資料偶有壞點(單日 ±90%),遮蔽後以前值補
    r = tlt_twd.pct_change()
    tlt_twd[r.abs() > 0.15] = float("nan")
    tlt_twd = tlt_twd.ffill()
    tlt_twd.to_frame(BOND).to_csv(cache)
prices_all[BOND] = tlt_twd.reindex(prices_all.index).ffill()

rev = fetch_month_revenue(list(config.UNIVERSE), start="2005-06-01",
                          cache_name="month_revenue_long.csv")
yoy_table = revenue_yoy_table(rev)

# ---------- 1. 週轉率統計 ----------
dates = prices_all.index
week_ends = pd.Series(dates, index=dates).resample("W-FRI").last().dropna()
week_ends = [d for d in week_ends if dates.get_loc(d) >= 130]

prev, changes, weeks_with_trade, filter_flips = None, 0, 0, 0
prev_filter = None
for d in week_ends:
    f = strategy.market_ok(bench, d)
    targets = set(strategy.target_holdings(prices, bench, d))
    if prev is not None:
        diff = len(prev - targets)          # 賣出檔數(=買進檔數,大致對稱)
        changes += diff
        if diff > 0:
            weeks_with_trade += 1
        if prev_filter is not None and f != prev_filter:
            filter_flips += 1
    prev, prev_filter = targets, f

n = len(week_ends) - 1
print("=== 週轉率統計(純動能策略,2007–2026,共 %d 週)===" % n)
print(f"平均每週更換:{changes/n:.2f} 檔(滿手 5 檔)")
print(f"完全沒交易的週:{(n-weeks_with_trade)/n:.0%}")
print(f"平均每檔持有:約 {5*n/changes:.1f} 週")
print(f"濾網切換(進出場)次數:{filter_flips} 次,平均每年 {filter_flips/19:.1f} 次")

# ---------- 2. 濾網轉債券 ----------
def decide_C(asof):
    return strategy.target_holdings(prices, bench, asof)

def decide_C_bond(asof):
    t = strategy.target_holdings(prices, bench, asof)
    return t if t else [BOND]

def decide_E(asof):
    return strategy.combined_targets(prices, bench, yoy_table, asof)

def decide_E_bond(asof):
    t = strategy.combined_targets(prices, bench, yoy_table, asof)
    return t if t else [BOND]

navs = {}
for name, fn in [("C_濾網抱現金", decide_C), ("C_濾網轉美債", decide_C_bond),
                 ("E_混合抱現金", decide_E), ("E_混合轉美債", decide_E_bond)]:
    navs[name] = simulate(prices_all, bench, fn)

print("\n=== 濾網觸發時:現金 vs 美債(2007–2026,含交易成本)===")
for name, nav in navs.items():
    s = stats(nav)
    print(f"{name:<10} 年化 {s['年化報酬']:>7} | 最大回檔 {s['最大回檔']:>7} | "
          f"夏普 {s['夏普值']} | 期末 {nav.iloc[-1]:>12,.0f}")

result = pd.DataFrame(navs)
yearly = result.resample("YE").last().pct_change(fill_method=None)
yearly.iloc[0] = result.resample("YE").last().iloc[0] / result.dropna().iloc[0] - 1
yearly.index = yearly.index.year
key_years = [2008, 2011, 2015, 2018, 2020, 2022, 2025]
print("\n=== 關鍵年度對照 ===")
print(yearly.loc[yearly.index.isin(key_years)].map(
    lambda x: f"{x:+.0%}" if pd.notna(x) else "").to_string())
