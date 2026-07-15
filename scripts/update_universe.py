# -*- coding: utf-8 -*-
"""投資範圍每季自動更新(解決「名單凍結」問題)

規則(兩個時鐘的慢時鐘):
  每季第一次執行時,從台灣證交所官方 OpenAPI 重建投資範圍——
  上市普通股中「市值最大的前 30 名」且日成交金額 > 1 億(流動性門檻)。
  市值 = 實收資本額/10(股數) × 收盤價。

  範圍的篩選標準是「市值+流動性」(慢,每季),不是強弱;
  強弱排名是策略的快時鐘(每週)。兩者分離,避免在小型股上跑動能。

非換季週執行會自動跳過;加 --force 可強制更新。
結果寫入 portfolio/universe.json,config.py 會優先讀取它。
"""
import json
import os
import sys
from datetime import date

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIVERSE_FILE = os.path.join(BASE, "portfolio", "universe.json")

TOP_N = 30                 # 市值前 30 名
MIN_TRADE_VALUE = 1e8      # 日成交金額門檻:1 億元

SECTOR = {"01": "水泥", "02": "食品", "03": "塑化", "04": "紡織", "05": "電機機械",
          "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙", "10": "鋼鐵",
          "11": "橡膠", "12": "汽車", "14": "建材營造", "15": "航運",
          "16": "觀光餐旅", "17": "金融", "18": "貿易百貨", "19": "綜合",
          "20": "其他", "21": "化學", "22": "生技醫療", "23": "油電燃氣",
          "24": "半導體", "25": "電腦硬體", "26": "光電", "27": "通信網路",
          "28": "電子零組件", "29": "電子通路", "30": "資訊服務", "31": "其他電子"}


def quarter(d):
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def main():
    force = "--force" in sys.argv
    today = date.today()
    if os.path.exists(UNIVERSE_FILE) and not force:
        with open(UNIVERSE_FILE, encoding="utf-8") as f:
            old = json.load(f)
        if quarter(date.fromisoformat(old["updated"])) == quarter(today):
            print(f"投資範圍本季({quarter(today)})已更新過,跳過")
            return

    print("從證交所 OpenAPI 重建投資範圍...")
    day = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
                       timeout=60).json()
    basic = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
                         timeout=60).json()
    info = {b["公司代號"]: b for b in basic}

    rows = []
    for r in day:
        code = r["Code"]
        if len(code) != 4 or code not in info:      # 僅上市普通股
            continue
        try:
            close = float(r["ClosingPrice"])
            value = float(r["TradeValue"])
            capital = float(info[code]["實收資本額"])
        except (ValueError, KeyError):
            continue
        if value < MIN_TRADE_VALUE or close <= 0:
            continue
        mcap = capital / 10 * close                 # 股數 × 收盤價
        rows.append((code, info[code]["公司簡稱"],
                     SECTOR.get(info[code]["產業別"], "其他"), mcap))

    rows.sort(key=lambda x: -x[3])
    top = rows[:TOP_N]
    universe = {f"{c}.TW": [name, sec] for c, name, sec, _ in top}

    old_set = set()
    if os.path.exists(UNIVERSE_FILE):
        with open(UNIVERSE_FILE, encoding="utf-8") as f:
            old_set = set(json.load(f)["universe"])
    else:
        sys.path.insert(0, BASE)
        import config
        old_set = set(config.UNIVERSE)

    with open(UNIVERSE_FILE, "w", encoding="utf-8") as f:
        json.dump({"updated": today.isoformat(),
                   "rule": f"上市普通股市值前 {TOP_N} 名,日成交金額>{MIN_TRADE_VALUE:,.0f}",
                   "universe": universe}, f, ensure_ascii=False, indent=2)

    new_set = set(universe)
    added = new_set - old_set
    removed = old_set - new_set
    print(f"投資範圍已更新({quarter(today)}):共 {len(universe)} 檔")
    if added:
        print("  新增:", "、".join(f"{universe[t][0]}({t.replace('.TW','')})" for t in sorted(added)))
    if removed:
        print("  移除:", "、".join(sorted(t.replace(".TW", "") for t in removed)))


if __name__ == "__main__":
    main()
