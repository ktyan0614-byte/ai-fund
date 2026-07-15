# -*- coding: utf-8 -*-
"""AI 虛擬投資公司 — 全域設定"""
import json as _json
import os as _os

# 投資範圍:台股各產業的大型權值股(流動性好、資料完整)
# 每季由 scripts/update_universe.py 依「市值前30名+流動性門檻」自動更新
# (寫入 portfolio/universe.json);以下固定名單僅為初始預設/備援。
# 格式: yfinance 代號: (中文名稱, 產業)
_DEFAULT_UNIVERSE = {
    "2330.TW": ("台積電", "半導體"),
    "2454.TW": ("聯發科", "半導體"),
    "2303.TW": ("聯電", "半導體"),
    "3711.TW": ("日月光投控", "半導體"),
    "2317.TW": ("鴻海", "電子製造"),
    "2382.TW": ("廣達", "電子製造"),
    "3231.TW": ("緯創", "電子製造"),
    "2357.TW": ("華碩", "電腦硬體"),
    "2376.TW": ("技嘉", "電腦硬體"),
    "2345.TW": ("智邦", "網通設備"),
    "2308.TW": ("台達電", "電子零組件"),
    "2327.TW": ("國巨", "電子零組件"),
    "3008.TW": ("大立光", "光學"),
    "2881.TW": ("富邦金", "金融"),
    "2882.TW": ("國泰金", "金融"),
    "2891.TW": ("中信金", "金融"),
    "2886.TW": ("兆豐金", "金融"),
    "1301.TW": ("台塑", "塑化"),
    "1101.TW": ("台泥", "水泥"),
    "2002.TW": ("中鋼", "鋼鐵"),
    "2603.TW": ("長榮", "航運"),
    "2609.TW": ("陽明", "航運"),
    "1216.TW": ("統一", "食品"),
    "2912.TW": ("統一超", "零售"),
    "2412.TW": ("中華電", "電信"),
    "2207.TW": ("和泰車", "汽車"),
}

_UNIVERSE_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                               "portfolio", "universe.json")
if _os.path.exists(_UNIVERSE_FILE):
    with open(_UNIVERSE_FILE, encoding="utf-8") as _f:
        UNIVERSE = {k: tuple(v) for k, v in _json.load(_f)["universe"].items()}
else:
    UNIVERSE = _DEFAULT_UNIVERSE

BENCHMARK = "0050.TW"          # 比較基準:元大台灣50 ETF
BENCHMARK_NAME = "元大台灣50 (0050)"

INITIAL_CASH = 100_000         # 初始資金(新台幣)

# --- 策略參數 ---
TOP_N = 5                      # 持有動能最強的前 N 檔
MOM_LOOKBACK = 60              # 動能回看天數(約一季的交易日)
MARKET_FILTER_MA = 120         # 大盤濾網:0050 收盤價低於 N 日均線時全數轉現金

# --- 美股 ETF 部門(帳戶四):被動持有 + 區間再平衡 ---
US_ETF_PORTFOLIO = {          # 代號: (名稱, 目標配置)
    "SPY": ("美國S&P500", 0.50),
    "QQQ": ("那斯達克100", 0.30),
    "VIG": ("美國股息成長", 0.20),
}
US_REBALANCE_BAND = 0.05      # 任一 ETF 偏離目標配置逾 5 個百分點才再平衡
US_FEE_RATE = 0.001           # 海外券商費率(約 0.1%,無證交稅,支援碎股)

# --- 交易成本(台股實際費率) ---
FEE_RATE = 0.001425 * 0.6      # 券商手續費 0.1425% × 常見電子下單 6 折
MIN_FEE = 1                    # 最低手續費 1 元(零股)
TAX_RATE = 0.003               # 證交稅 0.3%(僅賣出時課)
