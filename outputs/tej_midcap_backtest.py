# -*- coding: utf-8 -*-
"""中型股假設判決(依 outputs/midcap_preregistration.md 預先註冊的規格執行)

規格(已鎖定,不得調整):
  範圍:每季市值 31–150 名、月成交值>5億(point-in-time,含下市)
  策略:60 日動能(>0)、動能+營收各半、週頻、Y9997 跌破 120 日均線全出
  主測 Top10(Top5 僅參考);成本:手續費+證交稅+單邊 0.3% 滑價
  判準:Top10 夏普>0.83 且年化>+14.5% → 支持;僅夏普>0.83 → 部分支持;否則推翻
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEJ = os.path.join(BASE, "tej")

# ---------- 1. 合併與驗證中型股日資料 ----------
MERGED = os.path.join(TEJ, "prices_daily_midcap_merged.csv")
if not os.path.exists(MERGED):
    frames = []
    for f in sorted(glob.glob(os.path.join(TEJ, "prices_daily_midcap_2*"))):
        if f.startswith(os.path.join(TEJ, "~$")) or "~$" in f:
            continue
        df = pd.read_excel(f) if f.endswith(".xlsx") else \
            pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        assert list(df.columns) == ["代號", "名稱", "年月日", "報酬率％", "收盤價(元)"], \
            f"{os.path.basename(f)} 欄位不符"
        df.columns = ["code", "name", "date", "ret", "close"]
        df["code"] = df["code"].astype(str).str.strip()
        df["date"] = pd.to_datetime(df["date"].astype(str).str.replace("-", "/"),
                                    format="%Y/%m/%d")
        for c in ["ret", "close"]:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""),
                                  errors="coerce")
        frames.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,} 列 | {df['code'].nunique()} 檔")
    m = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"])
    # 品質檢查:報酬率 vs 收盤價
    bad = []
    for code, g in m.groupby("code"):
        if len(g) < 30:
            continue
        g = g.sort_values("date")
        implied = g["close"].pct_change(fill_method=None) * 100
        n_bad = ((implied - g["ret"]).abs() > 0.5).sum()
        if n_bad > len(g) * 0.02:
            bad.append((code, g["name"].iloc[0], n_bad, len(g)))
    print(f"合併 {len(m):,} 列、{m['code'].nunique()} 檔|"
          f"一致性異常: {len(bad)} 檔" + (f" {bad[:5]}" if bad else "(全過)"))
    m.to_csv(MERGED, index=False, encoding="utf-8-sig")

# ---------- 2. 組合完整面板(中型股 + 既有權值股/指數) ----------
mid = pd.read_csv(MERGED, encoding="utf-8-sig", parse_dates=["date"],
                  dtype={"code": str})
big = pd.read_csv(os.path.join(TEJ, "prices_daily_merged.csv"),
                  encoding="utf-8-sig", parse_dates=["date"], dtype={"code": str})
allp = pd.concat([big, mid], ignore_index=True).drop_duplicates(["code", "date"])
wide = allp.pivot(index="date", columns="code", values="close").sort_index()
last_valid = wide.apply(lambda s: s.last_valid_index())
px = wide.ffill(limit=15)
bench = px["Y9997"]
dates = px.index
print(f"完整面板:{wide.shape[1]} 檔 × {len(dates)} 日")

uh = json.load(open(os.path.join(TEJ, "universe_midcap.json"), encoding="utf-8"))
q_keys = sorted(uh["quarters"])
q_starts = pd.to_datetime(q_keys, format="%Y-%m")


def universe_at(d):
    i = q_starts.searchsorted(d, side="right") - 1
    return uh["quarters"][q_keys[i]] if i >= 0 else []


# ---------- 3. 營收(含發布日,缺值退法定期限) ----------
revs = []
for f in ["revenue_monthly.csv", "revenue_monthly_midcap.csv"]:
    r = pd.read_csv(os.path.join(TEJ, f), encoding="utf-8-sig",
                    dtype={"代號": str}, low_memory=False)
    r.columns = ["code", "name", "ym", "sales", "pub"]
    revs.append(r)
rev = pd.concat(revs, ignore_index=True)
rev["sales"] = pd.to_numeric(rev["sales"].astype(str).str.replace(",", ""),
                             errors="coerce")
rev["ym"] = pd.to_datetime(rev["ym"], format="%Y/%m")
rev["pub"] = pd.to_datetime(rev["pub"], format="%Y/%m/%d", errors="coerce")
rev["pub"] = rev["pub"].fillna(rev["ym"] + pd.offsets.MonthBegin(1)
                               + pd.Timedelta(days=9))
rev = rev.dropna(subset=["sales"]).drop_duplicates(["code", "ym"])
yoy_rows = {}
for code, g in rev.groupby("code"):
    g = g.sort_values("ym")
    y = g["sales"].values / pd.Series(g["sales"]).shift(12).values - 1
    s = pd.Series(y, index=g["pub"].values).replace([np.inf, -np.inf], np.nan)
    yoy_rows[code] = s.rolling(3).mean().groupby(level=0).last().sort_index()
yoy_table = pd.DataFrame(yoy_rows).sort_index()
print(f"營收:{rev['code'].nunique()} 檔")


def yoy_asof(asof):
    t = yoy_table.loc[:asof]
    return t.ffill().iloc[-1].dropna() if len(t) else pd.Series(dtype=float)


def mom_scores(asof, lookback=60):
    df = px.loc[:asof]
    uni = [c for c in universe_at(asof) if c in px.columns
           and last_valid[c] is not None
           and last_valid[c] >= asof - pd.Timedelta(days=5)]
    if len(df) < lookback + 1 or not uni:
        return pd.Series(dtype=float)
    ret = df[uni].iloc[-1] / df[uni].iloc[-lookback - 1] - 1
    return ret.dropna().sort_values(ascending=False)


def market_ok(asof, ma=120):
    s = bench.loc[:asof].dropna()
    return len(s) < ma or s.iloc[-1] >= s.iloc[-ma:].mean()


def decide(asof, w_mom=0.5, top_n=10):
    if not market_ok(asof):
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


FEE, TAX, SLIP = 0.001425 * 0.6, 0.003, 0.003   # 滑價 0.3%/邊(預先註冊)


def cost(v, sell):
    return max(1, v * FEE) + (v * TAX if sell else 0) + v * SLIP


def simulate(decide_fn, warmup=130):
    week_ends = pd.Series(dates, index=dates).resample("W-FRI").last().dropna()
    week_ends = set(d for d in week_ends if dates.get_loc(d) >= warmup)
    cash, shares, nav_rec = 100_000.0, {}, {}
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


navs = {}
s = bench.dropna()
navs["大盤含息"] = (s / s.iloc[130] * 100_000).iloc[130:]
for name, fn in [("E_中型Top10(主測)", lambda d: decide(d, 0.5, 10)),
                 ("C_中型Top10", lambda d: decide(d, 1.0, 10)),
                 ("E_中型Top5(參考)", lambda d: decide(d, 0.5, 5))]:
    print(f"回測 {name} ...")
    navs[name] = simulate(fn)

print("\n=== 中型股假設判決(2007–2026,含成本+0.3%滑價)===")
BENCH_SHARPE, BENCH_CAGR = None, None
for name, nav in navs.items():
    c, d, sh = stats(nav)
    if name == "大盤含息":
        BENCH_SHARPE, BENCH_CAGR = sh, c
    print(f"{name:<16} 年化 {c:+.1%} | 最大回檔 {d:.1%} | 夏普 {sh:.2f} | "
          f"期末 {nav.iloc[-1]:>12,.0f}")

c, d, sh = stats(navs["E_中型Top10(主測)"])
verdict = ("✅ 支持假設" if sh > BENCH_SHARPE and c > BENCH_CAGR else
           "🟡 部分支持(僅風險調整後優於大盤)" if sh > BENCH_SHARPE else
           "❌ 推翻假設")
print(f"\n預先註冊判準:夏普 {sh:.2f} vs 大盤 {BENCH_SHARPE:.2f}、"
      f"年化 {c:+.1%} vs {BENCH_CAGR:+.1%} → {verdict}")

res = pd.DataFrame(navs).dropna()
yearly = res.resample("YE").last().pct_change(fill_method=None)
yearly.iloc[0] = res.resample("YE").last().iloc[0] / res.iloc[0] - 1
yearly.index = yearly.index.year
print("\n=== 各年度報酬 ===")
print(yearly.map(lambda x: f"{x:+.0%}" if pd.notna(x) else "").to_string())

print("\n=== 滾動五年視窗 ===")
monthly = res.resample("ME").last()
for name in ["E_中型Top10(主測)", "大盤含息"]:
    m = monthly[name]
    r5 = ((m.shift(-60) / m) ** (1 / 5) - 1).dropna()
    print(f"{name:<16} 最差 {r5.min():+.1%} | 中位 {r5.median():+.1%} | "
          f"最佳 {r5.max():+.1%} | 五年虧損率 {(r5 < 0).mean():.0%}")
e5 = ((monthly["E_中型Top10(主測)"].shift(-60) / monthly["E_中型Top10(主測)"]) ** .2 - 1).dropna()
b5 = ((monthly["大盤含息"].shift(-60) / monthly["大盤含息"]) ** .2 - 1).dropna()
print(f"E 中型 Top10 五年視窗贏過大盤比例: {(e5 > b5).mean():.0%}")
