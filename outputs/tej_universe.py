# -*- coding: utf-8 -*-
"""用 TEJ 月市值(含已下市公司)重建 point-in-time 投資範圍

規則(對齊系統的動態名單規則):
  每季(1/4/7/10 月)換血一次,採用「前一個月底」的資料:
  上市普通股市值前 30 名,且月成交值 > 10 億(流動性門檻)。

輸出:
  tej/universe_history.json  每季的名單(point-in-time)
  並列印「歷史上曾入榜的全部股票」清單 → 用於 TEJ 第 3 步日資料下載
"""
import json
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEJ = os.path.join(BASE, "tej")


def load(name):
    df = pd.read_csv(os.path.join(TEJ, name), encoding="utf-8-sig",
                     low_memory=False)
    df.columns = ["code", "name", "ym", "mcap", "tval"]
    for c in ["mcap", "tval"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""),
                              errors="coerce")
    df["ym"] = pd.to_datetime(df["ym"], format="%Y/%m")
    df["code"] = df["code"].astype(str).str.strip()
    return df


live = load("mcap_monthly.csv")
dead = load("mcap_monthly_下市普通股.csv")
df = pd.concat([live, dead], ignore_index=True)
before = len(df)
df = df.drop_duplicates(["code", "ym"])
print(f"合併:現存 {live['code'].nunique()} 檔 + 下市 {dead['code'].nunique()} 檔 "
      f"= {df['code'].nunique()} 檔,{len(df)} 列(去重 {before - len(df)})")

names = df.sort_values("ym").groupby("code")["name"].last()

# 每季換血:1/4/7/10 月的名單,用前一個月底(3/6/9/12 月)的市值
TOP_N, MIN_TVAL = 30, 1_000    # 市值前30;月成交值>10億(單位:百萬元)
history = {}
snapshots = df[df["ym"].dt.month.isin([3, 6, 9, 12])]
for ym, g in snapshots.groupby("ym"):
    g = g.dropna(subset=["mcap"])
    g = g[g["tval"] > MIN_TVAL]
    top = g.nlargest(TOP_N, "mcap")
    eff = ym + pd.offsets.MonthBegin(1)          # 生效季度的第一個月
    history[eff.strftime("%Y-%m")] = list(top["code"])

ever = sorted({c for lst in history.values() for c in lst})
dead_codes = set(dead["code"])
gone = [c for c in ever if c in dead_codes]
current = set(load("mcap_monthly.csv")["code"])

print(f"\n季度名單數: {len(history)}(2005Q2 起)")
print(f"歷史上曾入榜: {len(ever)} 檔")
print(f"其中已下市: {len(gone)} 檔 → {['%s %s' % (c, names[c]) for c in gone]}")

with open(os.path.join(TEJ, "universe_history.json"), "w", encoding="utf-8") as f:
    json.dump({"rule": f"市值前{TOP_N}+月成交值>{MIN_TVAL}百萬",
               "names": {c: names[c] for c in ever},
               "quarters": history}, f, ensure_ascii=False, indent=1)

print(f"\n=== 第 3 步日資料下載清單(共 {len(ever)} 檔,另加指數 Y9997、Y9999)===")
for i in range(0, len(ever), 15):
    print(" ".join(ever[i:i + 15]))

# 抽查:幾個關鍵時點的名單
for q in ["2007-01", "2010-01", "2021-01", "2026-07"]:
    if q in history:
        print(f"\n{q} 名單: " + "、".join(f"{names[c]}" for c in history[q][:15]) + " ...")
