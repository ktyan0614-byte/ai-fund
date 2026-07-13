# -*- coding: utf-8 -*-
"""小資金規模測試:同一套策略在 1萬/3萬/10萬 下的表現差異"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config, strategy
from backtest import simulate, stats
from data import fetch_fields

tickers = list(config.UNIVERSE) + [config.BENCHMARK]
ohlc = fetch_fields(tickers, start="2020-01-01", cache_prefix="backtest")
prices_all = ohlc["Close"].ffill()
bench = prices_all[config.BENCHMARK]
prices = prices_all[list(config.UNIVERSE)]


def decide_C(asof):
    return strategy.target_holdings(prices, bench, asof)


def decide_A(asof):
    return [config.BENCHMARK]


print("同一套「動能+濾網」策略,不同資金規模(2020-2026 回測):")
for cash in [100_000, 30_000, 10_000]:
    nav = simulate(prices_all, bench, decide_C, initial_cash=cash)
    s = stats(nav)
    print(f"  初始 {cash:>7,} 元 | 年化 {s['年化報酬']} | 最大回檔 {s['最大回檔']} | 期末 {nav.iloc[-1]:>9,.0f}")

nav = simulate(prices_all, bench, decide_A, initial_cash=10_000)
s = stats(nav)
print("對照:同樣 1 萬元改買 0050 放著:")
print(f"  初始  10,000 元 | 年化 {s['年化報酬']} | 最大回檔 {s['最大回檔']} | 期末 {nav.iloc[-1]:>9,.0f}")
