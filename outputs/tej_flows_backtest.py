# -*- coding: utf-8 -*-
"""籌碼面假設判決(依 outputs/flows_preregistration.md 執行,不得調整)

主測:近 20 日外資買賣超金額合計 ÷ 最近月底市值,前 150 名單內排名 Top10 等權重
參考:同式投信。週頻、Y9997 120 日均線濾網、費稅+0.3% 滑價。
判準:夏普與年化皆勝同窗口大盤 → 支持;僅夏普 → 部分支持;否則推翻。
附註診斷(不影響判決):外資流訊號與 60 日動能的排名相關性。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEJ = os.path.join(BASE, "tej")

# ---------- 法人流 ----------
frames = []
for f in sorted(glob.glob(os.path.join(TEJ, "flows_daily_*"))):
    if f.endswith(".txt"):
        continue
    df = pd.read_excel(f) if f.endswith(".xlsx") else \
        pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
    df.columns = ["code", "name", "date", "for", "itc"]
    df["code"] = df["code"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    for c in ["for", "itc"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""),
                              errors="coerce")
    frames.append(df)
fl = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"])
print(f"法人流:{len(fl):,} 列|{fl['code'].nunique()} 檔|"
      f"{fl['date'].min().date()} ~ {fl['date'].max().date()}")
for_w = fl.pivot(index="date", columns="code", values="for").sort_index()
itc_w = fl.pivot(index="date", columns="code", values="itc").sort_index()
for_20 = for_w.fillna(0).rolling(20).sum()
itc_20 = itc_w.fillna(0).rolling(20).sum()

# ---------- 價格/市值/名單 ----------
mid = pd.read_csv(os.path.join(TEJ, "prices_daily_midcap_merged.csv"),
                  encoding="utf-8-sig", parse_dates=["date"], dtype={"code": str})
big = pd.read_csv(os.path.join(TEJ, "prices_daily_merged.csv"),
                  encoding="utf-8-sig", parse_dates=["date"], dtype={"code": str})
allp = pd.concat([big, mid], ignore_index=True).drop_duplicates(["code", "date"])
wide = allp.pivot(index="date", columns="code", values="close").sort_index()
last_valid = wide.apply(lambda s: s.last_valid_index())
px = wide.ffill(limit=15)
bench = px["Y9997"]
dates = px.index


def load_mcap(name):
    df = pd.read_csv(os.path.join(TEJ, name), encoding="utf-8-sig", low_memory=False)
    df.columns = ["code", "name", "ym", "mcap", "tval"]
    df["mcap"] = pd.to_numeric(df["mcap"].astype(str).str.replace(",", ""),
                               errors="coerce")
    df["ym"] = pd.to_datetime(df["ym"], format="%Y/%m") + pd.offsets.MonthEnd(0)
    df["code"] = df["code"].astype(str).str.strip()
    return df[["code", "ym", "mcap"]]


mc = pd.concat([load_mcap("mcap_monthly.csv"),
                load_mcap("mcap_monthly_下市普通股.csv")]).drop_duplicates(["code", "ym"])
mcap_w = mc.pivot(index="ym", columns="code", values="mcap").sort_index()

u1 = json.load(open(os.path.join(TEJ, "universe_history.json"), encoding="utf-8"))
u2 = json.load(open(os.path.join(TEJ, "universe_midcap.json"), encoding="utf-8"))
q_keys = sorted(set(u1["quarters"]) & set(u2["quarters"]))
quarters = {q: list(dict.fromkeys(u1["quarters"][q] + u2["quarters"][q]))
            for q in q_keys}
q_starts = pd.to_datetime(q_keys, format="%Y-%m")


def universe_at(d):
    i = q_starts.searchsorted(d, side="right") - 1
    return quarters[q_keys[i]] if i >= 0 else []


def signal_asof(asof, flow20):
    row = flow20.loc[:asof]
    if row.empty:
        return pd.Series(dtype=float)
    row = row.iloc[-1]
    mrow = mcap_w.loc[:asof]
    if mrow.empty:
        return pd.Series(dtype=float)
    mrow = mrow.ffill().iloc[-1]
    uni = [c for c in universe_at(asof) if c in row.index and c in mrow.index
           and c in px.columns and last_valid[c] is not None
           and last_valid[c] >= asof - pd.Timedelta(days=5)]
    sig = (row[uni] / mrow[uni]).replace([np.inf, -np.inf], np.nan).dropna()
    return sig.sort_values(ascending=False)


def market_ok(asof, ma=120):
    s = bench.loc[:asof].dropna()
    return len(s) < ma or s.iloc[-1] >= s.iloc[-ma:].mean()


def decide(asof, flow20, top_n=10):
    if not market_ok(asof):
        return []
    sig = signal_asof(asof, flow20)
    sig = sig[sig > 0]                      # 只買淨買超的
    return list(sig.head(top_n).index)


FEE, TAX, SLIP = 0.001425 * 0.6, 0.003, 0.003


def cost(v, sell):
    return max(1, v * FEE) + (v * TAX if sell else 0) + v * SLIP


def simulate(decide_fn, start_date):
    week_ends = pd.Series(dates, index=dates).resample("W-FRI").last().dropna()
    week_ends = set(d for d in week_ends if d >= start_date)
    cash, shares, nav_rec = 100_000.0, {}, {}
    for d in dates[dates.searchsorted(start_date):]:
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
                        n = min(int(diff // p[t]),
                                int(cash / (p[t] * (1 + FEE + SLIP))))
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
    return cagr, dd, r.mean() / r.std() * np.sqrt(252)


# 起點:流量資料起點 + 20 日窗 + 濾網暖身
start = max(fl["date"].min() + pd.Timedelta(days=240), dates[130])
start = dates[dates.searchsorted(start)]
print(f"回測窗口:{start.date()} ~ {dates[-1].date()}")

navs = {}
s = bench.loc[start:].dropna()
navs["大盤含息(同窗口)"] = s / s.iloc[0] * 100_000
for name, fl20 in [("F_外資流Top10(主測)", for_20), ("I_投信流Top10(參考)", itc_20)]:
    print(f"回測 {name} ...")
    navs[name] = simulate(lambda d, f=fl20: decide(d, f), start)

print("\n=== 籌碼面假設判決(含成本+0.3%滑價)===")
bs = stats(navs["大盤含息(同窗口)"])
for name, nav in navs.items():
    c, d, sh = stats(nav)
    print(f"{name:<18} 年化 {c:+.1%} | 最大回檔 {d:.1%} | 夏普 {sh:.2f} | "
          f"期末 {nav.iloc[-1]:>12,.0f}")

c, d, sh = stats(navs["F_外資流Top10(主測)"])
verdict = ("✅ 支持假設" if sh > bs[2] and c > bs[0] else
           "🟡 部分支持" if sh > bs[2] else "❌ 推翻假設")
print(f"\n判準:夏普 {sh:.2f} vs 大盤 {bs[2]:.2f}、年化 {c:+.1%} vs {bs[0]:+.1%}"
      f" → {verdict}")

res = pd.DataFrame(navs).dropna()
yearly = res.resample("YE").last().pct_change(fill_method=None)
yearly.iloc[0] = res.resample("YE").last().iloc[0] / res.iloc[0] - 1
yearly.index = yearly.index.year
print("\n=== 各年度報酬 ===")
print(yearly.map(lambda x: f"{x:+.0%}" if pd.notna(x) else "").to_string())

# 診斷:外資流 vs 60日動能 排名相關性(不影響判決)
corrs = []
week_ends = pd.Series(dates, index=dates).resample("W-FRI").last().dropna()
for d in [w for w in week_ends if w >= start][::4]:
    sig = signal_asof(d, for_20)
    if len(sig) < 30:
        continue
    dfp = px.loc[:d]
    mom = (dfp[sig.index].iloc[-1] / dfp[sig.index].iloc[-61] - 1).dropna()
    common = sig.index.intersection(mom.index)
    if len(common) > 30:
        corrs.append(sig[common].rank().corr(mom[common].rank()))
print(f"\n診斷:外資流訊號 vs 60日動能 排名相關性(週抽樣平均): {np.mean(corrs):.2f}")
