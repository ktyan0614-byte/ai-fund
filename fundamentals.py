# -*- coding: utf-8 -*-
"""基本面資料:台股月營收(FinMind 公開 API)

台股獨有優勢:上市公司依法每月 10 日前公佈上月營收,是全世界最即時的基本面數據。
本模組計算「月營收年增率(YoY)」——跟去年同月比,避開淡旺季干擾。

重要:回測時只能用「當時已公佈」的資料。營收在次月 10 日公佈,
所以本模組把每筆營收的「可用日」設為次月 10 日,決策日之前公佈的才拿來用。
"""
import os
import time

import pandas as pd
import requests

import config

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
API = "https://api.finmindtrade.com/api/v4/data"


def fetch_month_revenue(tickers, start="2018-06-01", cache_name=None):
    """抓多檔股票的月營收,回傳 long-format DataFrame。tickers 用 yfinance 代號。"""
    if cache_name:
        cache_path = os.path.join(CACHE_DIR, cache_name)
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, parse_dates=["avail_date"],
                               dtype={"stock_id": str})

    frames = []
    for t in tickers:
        sid = t.replace(".TW", "")
        r = requests.get(API, params={"dataset": "TaiwanStockMonthRevenue",
                                      "data_id": sid, "start_date": start},
                         timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            df = pd.DataFrame(data)
            df["ticker"] = t
            frames.append(df)
        time.sleep(0.4)   # 禮貌性間隔,避免觸發限流
    rev = pd.concat(frames, ignore_index=True)

    # date 欄位是「公佈月」的 1 號(如 2025-01-01 是 2024/12 的營收)
    # → 可用日設為該月 10 日(法定公佈期限)
    rev["avail_date"] = pd.to_datetime(rev["date"]) + pd.Timedelta(days=9)
    rev = rev[["ticker", "revenue_year", "revenue_month", "revenue", "avail_date"]]

    if cache_name:
        os.makedirs(CACHE_DIR, exist_ok=True)
        rev.to_csv(os.path.join(CACHE_DIR, cache_name), index=False)
    return rev


def revenue_yoy_table(rev: pd.DataFrame, smooth=3) -> pd.DataFrame:
    """轉成寬表:index=可用日, columns=ticker, 值=近 smooth 個月平均的營收年增率。

    用近 3 個月平均年增率而非單月:單月營收受出貨時點干擾大,平滑後訊號較穩。
    """
    out = {}
    for t, g in rev.groupby("ticker"):
        g = g.sort_values(["revenue_year", "revenue_month"]).drop_duplicates(
            ["revenue_year", "revenue_month"])
        yoy = g["revenue"].values / pd.Series(g["revenue"]).shift(12).values - 1
        s = pd.Series(yoy, index=g["avail_date"].values)
        s = s.replace([float("inf"), float("-inf")], float("nan"))  # 單月營收為 0 時
        out[t] = s.rolling(smooth).mean()
    table = pd.DataFrame(out).sort_index()
    return table


def yoy_asof(table: pd.DataFrame, asof) -> pd.Series:
    """取決策日當下「已公佈」的最新營收年增率(point-in-time)。"""
    avail = table.loc[:asof]
    if avail.empty:
        return pd.Series(dtype=float)
    return avail.ffill().iloc[-1].dropna()


# ---------- 季報(毛利率、獲利) ----------

FIN_TYPES = {"Revenue", "GrossProfit", "IncomeAfterTaxes"}

# 台股季報法定公佈期限:Q1→5/15、Q2→8/14、Q3→11/14、Q4(年報)→次年 3/31
_PUB_DEADLINE = {3: ("05-15", 0), 6: ("08-14", 0), 9: ("11-14", 0), 12: ("03-31", 1)}


def _pub_date(qend: pd.Timestamp) -> pd.Timestamp:
    mmdd, year_add = _PUB_DEADLINE[qend.month]
    return pd.Timestamp(f"{qend.year + year_add}-{mmdd}")


def fetch_financials(tickers, start="2005-01-01", cache_name=None):
    """抓季報損益表關鍵科目,回傳 long DataFrame(含保守的法定公佈日 avail_date)。"""
    if cache_name:
        cache_path = os.path.join(CACHE_DIR, cache_name)
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, parse_dates=["qend", "avail_date"])

    frames = []
    for t in tickers:
        sid = t.replace(".TW", "")
        r = requests.get(API, params={"dataset": "TaiwanStockFinancialStatements",
                                      "data_id": sid, "start_date": start},
                         timeout=60)
        r.raise_for_status()
        data = [row for row in r.json().get("data", []) if row["type"] in FIN_TYPES]
        if data:
            df = pd.DataFrame(data).pivot_table(index="date", columns="type",
                                                values="value", aggfunc="first")
            df["ticker"] = t
            frames.append(df.reset_index())
        time.sleep(0.4)
    fin = pd.concat(frames, ignore_index=True)
    fin["qend"] = pd.to_datetime(fin["date"])
    fin["avail_date"] = fin["qend"].map(_pub_date)
    fin = fin.drop(columns=["date"])

    if cache_name:
        os.makedirs(CACHE_DIR, exist_ok=True)
        fin.to_csv(os.path.join(CACHE_DIR, cache_name), index=False)
    return fin


def margin_trend_table(fin: pd.DataFrame) -> pd.DataFrame:
    """毛利率趨勢:本季毛利率 − 去年同季毛利率(百分點)。

    跟「去年同季」比而非上一季,避開淡旺季;比「變化量」而非水準,
    因為不同產業的毛利率水準天生不同(IC 設計 50% vs 代工 15%),
    但「毛利率在變好」對哪個產業都是好消息。
    """
    out = {}
    for t, g in fin.groupby("ticker"):
        g = g.sort_values("qend").drop_duplicates("qend")
        gm = g["GrossProfit"] / g["Revenue"]
        trend = gm.values - pd.Series(gm).shift(4).values
        out[t] = pd.Series(trend, index=g["avail_date"].values)
    return pd.DataFrame(out).sort_index()


def profitable_table(fin: pd.DataFrame) -> pd.DataFrame:
    """獲利門檻用:近四季稅後淨利合計(數值;NaN 表示資料不足,呼叫端應視為通過,
    只有明確知道虧損時才過濾——缺資料不等於壞公司)。"""
    out = {}
    for t, g in fin.groupby("ticker"):
        g = g.sort_values("qend").drop_duplicates("qend")
        ttm = pd.Series(g["IncomeAfterTaxes"].values).rolling(4).sum()
        out[t] = pd.Series(ttm.values, index=g["avail_date"].values)
    return pd.DataFrame(out).sort_index()
