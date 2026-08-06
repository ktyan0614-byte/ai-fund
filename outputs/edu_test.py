# -*- coding: utf-8 -*-
"""第 16–18 組教育性回測(依 outputs/edu_preregistration.md 規格執行)

16 低波動異常   :過去 250 日波動最低的 10 檔
17 等權 vs 市值 :全池 150 檔,唯一差別是權重
18 名字吉利度   :簡稱含吉利字者取市值最大 10 檔(刻意無經濟機制的雜訊對照)

共同:point-in-time 前150、季度再平衡、等權(除市值加權組)、含成本+滑價、無濾網。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEJ = os.path.join(BASE, "tej")
sys.path.insert(0, BASE)

FEE, TAX, SLIP = 0.001425 * 0.6, 0.003, 0.003
LUCKY = "金富利鴻隆興順旺鑫泰昌發"

# ---------- 資料 ----------
big = pd.read_csv(os.path.join(TEJ, "prices_daily_merged.csv"),
                  encoding="utf-8-sig", parse_dates=["date"], dtype={"code": str})
mid = pd.read_csv(os.path.join(TEJ, "prices_daily_midcap_merged.csv"),
                  encoding="utf-8-sig", parse_dates=["date"], dtype={"code": str})
allp = pd.concat([big, mid], ignore_index=True).drop_duplicates(["code", "date"])
wide = allp.pivot(index="date", columns="code", values="close").sort_index()
last_valid = wide.apply(lambda s: s.last_valid_index())
px = wide.ffill(limit=15)
bench = px["Y9997"]
dates = px.index

u1 = json.load(open(os.path.join(TEJ, "universe_history.json"), encoding="utf-8"))
u2 = json.load(open(os.path.join(TEJ, "universe_midcap.json"), encoding="utf-8"))
q_keys = sorted(set(u1["quarters"]) & set(u2["quarters"]))
quarters = {q: list(dict.fromkeys(u1["quarters"][q] + u2["quarters"][q]))
            for q in q_keys}
q_starts = pd.to_datetime(q_keys, format="%Y-%m")
names = {**u1["names"], **u2["names"]}


def load_mcap():
    fr = []
    for f in ["mcap_monthly.csv", "mcap_monthly_下市普通股.csv"]:
        d = pd.read_csv(os.path.join(TEJ, f), encoding="utf-8-sig", low_memory=False)
        d.columns = ["code", "name", "ym", "mcap", "tval"]
        d["mcap"] = pd.to_numeric(d["mcap"].astype(str).str.replace(",", ""),
                                  errors="coerce")
        d["ym"] = pd.to_datetime(d["ym"], format="%Y/%m") + pd.offsets.MonthEnd(0)
        d["code"] = d["code"].astype(str).str.strip()
        fr.append(d[["code", "ym", "mcap"]])
    return pd.concat(fr).drop_duplicates(["code", "ym"]).pivot(
        index="ym", columns="code", values="mcap").sort_index()


mcap_w = load_mcap()
print(f"資料:{wide.shape[1]} 檔 × {len(dates)} 日|名單 {len(q_keys)} 季")


def universe_at(d):
    i = q_starts.searchsorted(d, side="right") - 1
    if i < 0:
        return []
    return [c for c in quarters[q_keys[i]]
            if c in px.columns and last_valid[c] is not None
            and last_valid[c] >= d - pd.Timedelta(days=5)]


def mcap_asof(d):
    m = mcap_w.loc[:d]
    return m.ffill().iloc[-1] if len(m) else pd.Series(dtype=float)


# ---------- 三組選股規則 ----------
def pick_lowvol(d, n=10):
    uni = universe_at(d)
    win = px.loc[:d, uni].tail(251)
    if len(win) < 200:
        return []
    vol = win.pct_change(fill_method=None).std().dropna()
    return list(vol.nsmallest(n).index)


def pick_lucky(d, n=10):
    uni = universe_at(d)
    cand = [c for c in uni if any(ch in names.get(c, "") for ch in LUCKY)]
    if not cand:
        return []
    m = mcap_asof(d).reindex(cand).dropna()
    return list(m.nlargest(min(n, len(m))).index)


def pick_all(d):
    return universe_at(d)


# ---------- 模擬(季度再平衡) ----------
def simulate(pick_fn, cap_weight=False, warmup=260):
    q_ends = pd.Series(dates, index=dates).resample("QE").last().dropna()
    q_ends = [d for d in q_ends if dates.get_loc(d) >= warmup]
    if not q_ends:
        return pd.Series(dtype=float)
    cash, shares, rec, delisted = 100_000.0, {}, {}, []
    reb = set(q_ends)
    for d in dates[dates.get_loc(q_ends[0]):]:
        p = px.loc[d]
        if d in reb:
            tg = [t for t in pick_fn(d) if not np.isnan(p.get(t, np.nan))]
            # 持股已無報價(下市/合併/停牌)→ 以最後有效價變現。
            # 季頻再平衡有三個月空窗,不處理的話 nav 會變 NaN、組合永久凍結。
            for t in list(shares):
                if shares[t] > 0 and np.isnan(p.get(t, np.nan)):
                    lastp = wide[t].loc[:d].dropna()
                    v = shares[t] * (lastp.iloc[-1] if len(lastp) else 0)
                    if v > 0:
                        cash += v - (max(1, v * FEE) + v * TAX + v * SLIP)
                    delisted.append(names.get(t, t))
                    shares[t] = 0
            for t in list(shares):
                if t not in tg and shares[t] > 0 and not np.isnan(p[t]):
                    v = shares[t] * p[t]
                    cash += v - (max(1, v * FEE) + v * TAX + v * SLIP)
                    shares[t] = 0
            if tg:
                nav = cash + sum(n * p[t] for t, n in shares.items()
                                 if n > 0 and not np.isnan(p[t]))
                if cap_weight:
                    m = mcap_asof(d).reindex(tg).fillna(0)
                    w = (m / m.sum()) if m.sum() > 0 else pd.Series(1 / len(tg), index=tg)
                else:
                    w = pd.Series(1 / len(tg), index=tg)
                for t in tg:
                    diff = nav * w[t] - shares.get(t, 0) * p[t]
                    if diff > p[t]:
                        n = min(int(diff // p[t]),
                                int(cash / (p[t] * (1 + FEE + SLIP))))
                        if n > 0:
                            v = n * p[t]
                            shares[t] = shares.get(t, 0) + n
                            cash -= v + max(1, v * FEE) + v * SLIP
                    elif diff < -p[t] and shares.get(t, 0) > 0:
                        n = min(int(-diff // p[t]), shares[t])
                        if n > 0:
                            v = n * p[t]
                            cash += v - (max(1, v * FEE) + v * TAX + v * SLIP)
                            shares[t] -= n
        rec[d] = cash + sum(n * p[t] for t, n in shares.items()
                            if n > 0 and not np.isnan(p[t]))
    if delisted:
        print(f"    (期間 {len(delisted)} 次持股失去報價已變現: {set(delisted)})")
    return pd.Series(rec)


def stats(nav):
    nav = nav.dropna()
    y = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1
    dd = (nav / nav.cummax() - 1).min()
    r = nav.pct_change(fill_method=None).dropna()
    return c, dd, r.mean() / r.std() * np.sqrt(252)


runs = {}
print("回測 16 低波動 ...");      runs["16 低波動Top10"] = simulate(pick_lowvol)
print("回測 17a 全池等權 ...");   runs["17a 全池等權"] = simulate(pick_all, False)
print("回測 17b 全池市值加權 ..."); runs["17b 全池市值加權"] = simulate(pick_all, True)
print("回測 18 名字吉利 ...");    runs["18 名字吉利Top10"] = simulate(pick_lucky)

b = bench.reindex(list(runs.values())[0].index).ffill()
runs["基準 大盤含息"] = b / b.iloc[0] * 100_000

print(f"\n=== 第 16–18 組結果(2007–2026,季度再平衡,含成本+0.3%滑價)===")
res = {}
for k, nav in runs.items():
    c, dd, sh = stats(nav)
    res[k] = (c, dd, sh)
    print(f"{k:<16} 年化 {c:+6.1%} | 最大回檔 {dd:6.1%} | 夏普 {sh:.2f}")

bs = res["基準 大盤含息"]
print(f"\n判準對照(夏普 vs 大盤 {bs[2]:.2f}):")
for k in ["16 低波動Top10", "18 名字吉利Top10"]:
    v = "✅ 高於大盤" if res[k][2] > bs[2] else "❌ 未高於大盤"
    print(f"  {k}: 夏普 {res[k][2]:.2f} → {v}")
print(f"\n17 加權方式差距: 等權 {res['17a 全池等權'][0]:+.1%} vs "
      f"市值加權 {res['17b 全池市值加權'][0]:+.1%} "
      f"(差 {(res['17a 全池等權'][0]-res['17b 全池市值加權'][0])*100:+.1f} 個百分點/年)")

df = pd.DataFrame(runs).dropna()
yr = df.resample("YE").last().pct_change(fill_method=None)
yr.iloc[0] = df.resample("YE").last().iloc[0] / df.iloc[0] - 1
yr.index = yr.index.year
print("\n=== 各年度報酬 ===")
print(yr.map(lambda x: f"{x:+.0%}" if pd.notna(x) else "").to_string())

# 名字吉利組抽查
print("\n名字吉利組近期選中:", [names.get(c, c) for c in pick_lucky(dates[-1])])
