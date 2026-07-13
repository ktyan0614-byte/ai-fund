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
