# -*- coding: utf-8 -*-
"""TEJ 無偏總回測(point-in-time 動態名單,含已下市公司)

相對於舊回測的三個修正:
  1. 投資範圍:每季用「當時」的市值前 30 名(含後來下市的公司)
  2. 大盤基準與濾網:加權報酬指數 Y9997(含息)——不再低估基準
  3. 營收訊號:用 TEJ 真實「營收發布日」,不再用法定期限假設

輸出:總表、逐年表、Top-N 掃描、濾網參數掃描、滾動五年視窗分佈
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEJ = os.path.join(BASE, "tej")
sys.path.insert(0, BASE)
import config

FEE, TAX, MIN_FEE = config.FEE_RATE, config.TAX_RATE, config.MIN_FEE
CASH0 = 100_000

# ---------- 資料 ----------
full = pd.read_csv(os.path.join(TEJ, "prices_daily_merged.csv"),
                   encoding="utf-8-sig", parse_dates=["date"],
                   dtype={"code": str})
wide = full.pivot(index="date", columns="code", values="close").sort_index()
last_valid = wide.apply(lambda s: s.last_valid_index())
px = wide.ffill(limit=15)
bench = px["Y9997"]                      # 加權報酬指數(含息)
dates = px.index

uh = json.load(open(os.path.join(TEJ, "universe_history.json"), encoding="utf-8"))
q_keys = sorted(uh["quarters"])
q_starts = pd.to_datetime(q_keys, format="%Y-%m")


def universe_at(d):
    i = q_starts.searchsorted(d, side="right") - 1
    return uh["quarters"][q_keys[i]] if i >= 0 else []


rev = pd.read_csv(os.path.join(TEJ, "revenue_monthly.csv"), encoding="utf-8-sig",
                  dtype={"代號": str}, low_memory=False)
rev.columns = ["code", "name", "ym", "sales", "pub"]
rev["sales"] = pd.to_numeric(rev["sales"].astype(str).str.replace(",", ""),
                             errors="coerce")
rev["ym"] = pd.to_datetime(rev["ym"], format="%Y/%m")
rev["pub"] = pd.to_datetime(rev["pub"], format="%Y/%m/%d", errors="coerce")
# TEJ 發布日欄位 2013 年起才有;缺值時用法定期限(次月 10 日)保守備援
fallback = rev["ym"] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)
rev["pub"] = rev["pub"].fillna(fallback)
rev = rev.dropna(subset=["sales"])

yoy_rows = {}
for code, g in rev.groupby("code"):
    g = g.sort_values("ym").drop_duplicates("ym")
    y = g["sales"].values / pd.Series(g["sales"]).shift(12).values - 1
    s = pd.Series(y, index=g["pub"].values).replace([np.inf, -np.inf], np.nan)
    s = s.rolling(3).mean()
    # 同日發布多個月營收(補發)時,取最新月份的值
    yoy_rows[code] = s.groupby(level=0).last().sort_index()
yoy_table = pd.DataFrame(yoy_rows).sort_index()
print(f"資料就緒:{wide.shape[1]} 檔 × {len(dates)} 日|"
      f"名單 {len(q_keys)} 季|營收 {rev['code'].nunique()} 檔(真實發布日)")


def yoy_asof(asof):
    t = yoy_table.loc[:asof]
    return t.ffill().iloc[-1].dropna() if len(t) else pd.Series(dtype=float)


def mom_scores(asof, lookback=60):
    df = px.loc[:asof]
    if len(df) < lookback + 1:
        return pd.Series(dtype=float)
    uni = [c for c in universe_at(asof) if c in px.columns
           and last_valid[c] is not None
           and last_valid[c] >= asof - pd.Timedelta(days=5)]   # 排除停止交易者
    ret = df[uni].iloc[-1] / df[uni].iloc[-lookback - 1] - 1
    return ret.dropna().sort_values(ascending=False)


def market_ok(asof, ma):
    s = bench.loc[:asof].dropna()
    return len(s) < ma or s.iloc[-1] >= s.iloc[-ma:].mean()


def decide(asof, w_mom=0.5, top_n=5, ma=120):
    if not market_ok(asof, ma):
        return []
    mom = mom_scores(asof)
    mom = mom[mom > 0]
    if mom.empty:
        return []
    if w_mom >= 1.0:
        return list(mom.head(top_n).index)
    f = yoy_asof(asof).reindex(mom.index).dropna()
    common = mom.index.intersection(f.index)
    if len(common) == 0:
        return list(mom.head(top_n).index)
    combo = w_mom * mom[common].rank(pct=True) + (1 - w_mom) * f[common].rank(pct=True)
    return list(combo.sort_values(ascending=False).head(top_n).index)


def cost(v, sell):
    return max(MIN_FEE, v * FEE) + (v * TAX if sell else 0)


def simulate(decide_fn, warmup=130):
    week_ends = pd.Series(dates, index=dates).resample("W-FRI").last().dropna()
    week_ends = set(d for d in week_ends if dates.get_loc(d) >= warmup)
    cash, shares, nav_rec = float(CASH0), {}, {}
    start_i = min(dates.get_loc(d) for d in week_ends)
    for d in dates[start_i:]:
        p = px.loc[d]
        if d in week_ends:
            targets = [t for t in decide_fn(d) if not np.isnan(p.get(t, np.nan))]
            for t in list(shares):
                if t not in targets and shares[t] > 0 and not np.isnan(p[t]):
                    v = shares[t] * p[t]
                    cash += v - cost(v, True)
                    shares[t] = 0
            if targets:
                nav_now = cash + sum(n * p[t] for t, n in shares.items() if n > 0)
                per = nav_now / len(targets)
                for t in targets:
                    diff = per - shares.get(t, 0) * p[t]
                    if diff > p[t]:
                        n = min(int(diff // p[t]), int(cash / (p[t] * (1 + FEE))))
                        if n > 0:
                            v = n * p[t]
                            shares[t] = shares.get(t, 0) + n
                            cash -= v + cost(v, False)
        nav_rec[d] = cash + sum(n * p[t] for t, n in shares.items()
                                if n > 0 and not np.isnan(p[t]))
    return pd.Series(nav_rec)


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    r = nav.pct_change(fill_method=None).dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252)
    return cagr, dd, sharpe


# ---------- 主回測 ----------
runs = {
    "大盤含息(Y9997)": None,
    "C_純動能Top5": lambda d: decide(d, w_mom=1.0),
    "E_動能+營收Top5": lambda d: decide(d, w_mom=0.5),
}
navs = {}
for name, fn in runs.items():
    if fn is None:
        s = bench.dropna()
        navs[name] = s / s.iloc[130] * CASH0
        navs[name] = navs[name].loc[navs[name].index[130]:]
    else:
        print(f"回測 {name} ...")
        navs[name] = simulate(fn)

print("\n=== 無偏總回測(動態名單+含下市公司,2007–2026,含成本)===")
for name, nav in navs.items():
    c, d, sh = stats(nav)
    print(f"{name:<14} 年化 {c:+.1%} | 最大回檔 {d:.1%} | 夏普 {sh:.2f} | "
          f"期末 {nav.iloc[-1]:>12,.0f}")

res = pd.DataFrame(navs).dropna()
yearly = res.resample("YE").last().pct_change(fill_method=None)
yearly.iloc[0] = res.resample("YE").last().iloc[0] / res.iloc[0] - 1
yearly.index = yearly.index.year
print("\n=== 各年度報酬 ===")
print(yearly.map(lambda x: f"{x:+.0%}" if pd.notna(x) else "").to_string())

# ---------- Top-N 掃描 ----------
print("\n=== Top-N 掃描(動能+營收)===")
for n in [3, 5, 8, 10]:
    nav = simulate(lambda d, n=n: decide(d, w_mom=0.5, top_n=n))
    c, dd, sh = stats(nav)
    print(f"  Top{n:>2}: 年化 {c:+.1%} | 最大回檔 {dd:.1%} | 夏普 {sh:.2f}")
    if n == 8:
        navs["E_Top8"] = nav

# ---------- 濾網參數掃描 ----------
print("\n=== 濾網均線掃描(動能+營收 Top5)===")
for ma in [100, 120, 150, 200]:
    nav = simulate(lambda d, ma=ma: decide(d, w_mom=0.5, ma=ma))
    c, dd, sh = stats(nav)
    print(f"  {ma:>3} 日: 年化 {c:+.1%} | 最大回檔 {dd:.1%} | 夏普 {sh:.2f}")

# ---------- 滾動五年視窗 ----------
print("\n=== 滾動五年視窗(每月起算,年化報酬分佈)===")
monthly = pd.DataFrame(navs).dropna().resample("ME").last()
win = 60
roll = {}
for name in ["E_動能+營收Top5", "C_純動能Top5", "大盤含息(Y9997)"]:
    m = monthly[name]
    cagr5 = (m.shift(-win) / m) ** (1 / 5) - 1
    roll[name] = cagr5.dropna()
    r = roll[name]
    print(f"{name:<14} 視窗數 {len(r)} | 最差 {r.min():+.1%} | 25分位 {r.quantile(.25):+.1%}"
          f" | 中位 {r.median():+.1%} | 75分位 {r.quantile(.75):+.1%} | 最佳 {r.max():+.1%}")
beat = (roll["E_動能+營收Top5"] > roll["大盤含息(Y9997)"]).mean()
neg = (roll["E_動能+營收Top5"] < 0).mean()
print(f"\nE 策略五年視窗:贏過大盤的比例 {beat:.0%} | 五年下來仍虧損的比例 {neg:.0%}")
worst_start = roll["E_動能+營收Top5"].idxmin()
print(f"最差視窗起點:{worst_start.strftime('%Y-%m')}(年化 {roll['E_動能+營收Top5'].min():+.1%})")

print("\n=== 逐年起算的五年視窗(1 月起算)===")
print(f"{'起點':<8}{'E 動能+營收':>10}{'C 純動能':>10}{'大盤含息':>10}")
for y in range(2007, 2022):
    key = pd.Timestamp(f"{y}-01-31")
    row = []
    for name in ["E_動能+營收Top5", "C_純動能Top5", "大盤含息(Y9997)"]:
        r = roll[name]
        idx = r.index[r.index.searchsorted(key)] if r.index.searchsorted(key) < len(r) else None
        row.append(f"{r.loc[idx]:+.1%}" if idx is not None and abs((idx-key).days) < 45 else "—")
    print(f"{y:<10}{row[0]:>10}{row[1]:>10}{row[2]:>10}")
