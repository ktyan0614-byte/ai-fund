# -*- coding: utf-8 -*-
"""0050正2(00631L)每日再平衡 2x 槓桿的長期回測與滾動視窗

00631L 真實資料只到 2015,故以加權報酬指數(Y9997,含息)模擬每日 2x,
扣年化成本(內扣費用+借券≈1.6%/年),再用真實 00631L 校準模擬誤差。
重點:每日再平衡 2x ≠ 長期報酬 2x —— 波動耗損(volatility decay)。
"""
import os, sys
import numpy as np
import pandas as pd
import yfinance as yf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
full = pd.read_csv(os.path.join(BASE, "tej", "prices_daily_merged.csv"),
                   encoding="utf-8-sig", parse_dates=["date"], dtype={"code": str})
bench = full[full["code"] == "Y9997"].set_index("date")["close"].sort_index()
r = bench.pct_change(fill_method=None).dropna()          # 大盤含息日報酬

COST_Y = 0.016                                            # 年化成本(費用+借券)
cost_d = COST_Y / 252


def build(mult):
    daily = mult * r - (cost_d if mult != 1 else 0)      # 每日 2x 再平衡 - 成本
    return (1 + daily).cumprod()


nav1 = build(1)      # 大盤含息(原型)
nav2 = build(2)      # 每日 2x

# --- 用真實 00631L 校準 2015 起 ---
real = yf.download("00631L.TW", start="2015-01-05", auto_adjust=True,
                   progress=False)["Close"].dropna()
if hasattr(real, "columns"):
    real = real.iloc[:, 0]
sim2 = nav2.reindex(real.index).ffill()
sim2 = sim2 / sim2.iloc[0]
real_n = real / real.iloc[0]
err = (sim2.iloc[-1] / real_n.iloc[-1] - 1)
print(f"模擬 vs 真實 00631L(2015–2026 期末值誤差): {err:+.1%}"
      f"  → 模擬{'偏高' if err>0 else '偏低'},足以看趨勢")


def stats(nav):
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    d = nav.pct_change(fill_method=None).dropna()
    return cagr, dd, d.mean() / d.std() * np.sqrt(252)


print("\n=== 19 年回測(2007–2026,模擬,含成本)===")
for name, nav in [("大盤含息(1x)", nav1), ("0050正2(每日2x)", nav2)]:
    c, dd, sh = stats(nav)
    print(f"{name:<16} 年化 {c:+.1%} | 最大回檔 {dd:.1%} | 夏普 {sh:.2f} | "
          f"期末 {nav.iloc[-1]/nav1.iloc[0]*100000:>12,.0f}")

# 各年度
res = pd.DataFrame({"1x": nav1, "2x": nav2}).dropna()
yearly = res.resample("YE").last().pct_change(fill_method=None)
yearly.iloc[0] = res.resample("YE").last().iloc[0] / res.iloc[0] - 1
yearly.index = yearly.index.year
print("\n=== 各年度報酬 ===")
print(yearly.map(lambda x: f"{x:+.0%}" if pd.notna(x) else "").to_string())

# 滾動五年
print("\n=== 滾動五年視窗(每月起算)===")
m = res.resample("ME").last()
for name in ["1x", "2x"]:
    s = m[name]
    r5 = ((s.shift(-60) / s) ** (1 / 5) - 1).dropna()
    lbl = "大盤含息" if name == "1x" else "0050正2"
    print(f"{lbl:<10} 最差 {r5.min():+.1%} | 25% {r5.quantile(.25):+.1%} | "
          f"中位 {r5.median():+.1%} | 75% {r5.quantile(.75):+.1%} | "
          f"最佳 {r5.max():+.1%} | 五年虧損率 {(r5<0).mean():.0%}")

r5_2 = ((m["2x"].shift(-60) / m["2x"]) ** .2 - 1).dropna()
r5_1 = ((m["1x"].shift(-60) / m["1x"]) ** .2 - 1).dropna()
print(f"\n0050正2 五年視窗贏過 1x 的比例: {(r5_2 > r5_1).mean():.0%}")
# 最差起點對照
worst = r5_2.idxmin()
print(f"2x 最差五年起點: {worst.strftime('%Y-%m')} "
      f"(2x {r5_2.loc[worst]:+.1%} vs 同期 1x {r5_1.loc[worst]:+.1%})")
