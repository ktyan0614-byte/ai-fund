# -*- coding: utf-8 -*-
"""股價資料下載(yfinance),含簡單快取"""
import os
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def fetch_prices(tickers, start, end=None, cache_name=None):
    """下載多檔股票的調整後收盤價,回傳 DataFrame(index=日期, columns=代號)。"""
    if cache_name:
        cache_path = os.path.join(CACHE_DIR, cache_name)
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=list(tickers)[0])
    close = close.dropna(how="all")

    if cache_name:
        os.makedirs(CACHE_DIR, exist_ok=True)
        close.to_csv(os.path.join(CACHE_DIR, cache_name))
    return close


def fetch_fields(tickers, start, end=None,
                 fields=("Open", "High", "Low", "Close"), cache_prefix=None):
    """一次下載開高低收,回傳 {欄位: DataFrame}。皆為還原(除權息調整)價。"""
    if cache_prefix:
        paths = {f: os.path.join(CACHE_DIR, f"{cache_prefix}_{f}.csv") for f in fields}
        if all(os.path.exists(p) for p in paths.values()):
            return {f: pd.read_csv(p, index_col=0, parse_dates=True)
                    for f, p in paths.items()}

    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False)
    out = {}
    for f in fields:
        df = raw[f]
        if isinstance(df, pd.Series):
            df = df.to_frame(name=list(tickers)[0])
        out[f] = df.dropna(how="all")

    if cache_prefix:
        os.makedirs(CACHE_DIR, exist_ok=True)
        for f in fields:
            out[f].to_csv(os.path.join(CACHE_DIR, f"{cache_prefix}_{f}.csv"))
    return out
