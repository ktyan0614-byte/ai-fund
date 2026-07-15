# -*- coding: utf-8 -*-
"""合併 TEJ 分批日資料並做完整品質檢查

檢查項目:
  1. 各批欄位一致、批與批之間無縫(交易日連續、無重複)
  2. 每檔股票「報酬率累乘」vs「調整收盤價」一致性(資料壞點偵測)
  3. 股票數隨年代變化(上市/下市的自然增減)
輸出:
  tej/prices_daily_merged.csv  合併後長表
  tej/px_close_wide.csv        寬表(index=日期, columns=代號)→ 回測引擎直接用
"""
import glob
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEJ = os.path.join(BASE, "tej")

files = sorted(f for f in glob.glob(os.path.join(TEJ, "prices_daily_2*.csv"))
               if "merged" not in f)
print(f"找到 {len(files)} 批檔案")

frames = []
for f in files:
    df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
    assert list(df.columns) == ["代號", "名稱", "年月日", "報酬率％", "收盤價(元)"], \
        f"{os.path.basename(f)} 欄位不符: {list(df.columns)}"
    df.columns = ["code", "name", "date", "ret", "close"]
    df["code"] = df["code"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    for c in ["ret", "close"]:      # 千元股有千分位逗號,先去除再轉數字
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""),
                              errors="coerce")
    frames.append(df)
    print(f"  {os.path.basename(f)}: {len(df):,} 列 | "
          f"{df['date'].min().date()} ~ {df['date'].max().date()} | "
          f"{df['code'].nunique()} 檔")

full = pd.concat(frames, ignore_index=True)
dup = full.duplicated(["code", "date"]).sum()
full = full.drop_duplicates(["code", "date"]).sort_values(["code", "date"])
print(f"\n合併: {len(full):,} 列 | {full['code'].nunique()} 檔 | 重複列: {dup}")

# --- 檢查 1:批間交易日連續性(用台積電當探針) ---
tsmc = full[full["code"] == "2330"].set_index("date").sort_index()
gaps = tsmc.index.to_series().diff().dt.days
big_gaps = gaps[gaps > 11]      # 台股連假最長約 9-10 天
print(f"批間斷層檢查(>11天無交易日): {len(big_gaps)} 處"
      + ("" if big_gaps.empty else f" → {list(big_gaps.index.date)}"))

# --- 檢查 2:報酬率 vs 收盤價一致性(全部股票) ---
bad_stats = []
for code, g in full.groupby("code"):
    if code.startswith("Y") or len(g) < 30:
        continue
    g = g.sort_values("date")
    implied = g["close"].pct_change(fill_method=None) * 100
    bad = ((implied - g["ret"]).abs() > 0.5).sum()   # 容忍小數化誤差
    if bad > len(g) * 0.02:                          # 超過 2% 天數不吻合才報警
        bad_stats.append((code, g["name"].iloc[0], bad, len(g)))
print(f"報酬率一致性: {'全部通過' if not bad_stats else '異常股票如下'}")
for c, n, b, tot in bad_stats[:10]:
    print(f"  ⚠ {c} {n}: {b}/{tot} 天不吻合")

# --- 檢查 3:每年股票數 ---
per_year = full[~full["code"].str.startswith("Y")].groupby(
    full["date"].dt.year)["code"].nunique()
print("\n每年有資料的股票數:")
print(per_year.to_string())

# --- 輸出 ---
full.to_csv(os.path.join(TEJ, "prices_daily_merged.csv"), index=False,
            encoding="utf-8-sig")
wide = full.pivot(index="date", columns="code", values="close").sort_index()
wide.to_csv(os.path.join(TEJ, "px_close_wide.csv"), encoding="utf-8-sig")
print(f"\n已輸出 merged({len(full):,} 列)與寬表({wide.shape[0]} 日 × {wide.shape[1]} 檔)")
