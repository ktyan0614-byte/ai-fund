# -*- coding: utf-8 -*-
"""策略邏輯:動能選股 + 大盤濾網

白話說明:
1. 動能選股 — 「最近一季漲得最好的股票,短期內通常會繼續強」(學術上稱動能效應)。
   每週把投資範圍內的股票依「過去 60 個交易日的報酬率」排序,買進前 5 名、等權重。
2. 大盤濾網 — 「大盤趨勢往下時,再強的股票也容易被拖累」。
   當 0050 收盤價跌破 120 日均線,代表整體市場轉弱,全部賣出改抱現金,等趨勢回穩再進場。
"""
import pandas as pd
import config


def momentum_scores(prices: pd.DataFrame, asof=None, lookback=None) -> pd.Series:
    """計算各股票過去 lookback 日的報酬率(動能分數),由高到低排序。"""
    lookback = lookback or config.MOM_LOOKBACK
    df = prices.loc[:asof] if asof is not None else prices
    if len(df) < lookback + 1:
        return pd.Series(dtype=float)
    ret = df.iloc[-1] / df.iloc[-lookback - 1] - 1
    return ret.dropna().sort_values(ascending=False)


def market_ok(bench: pd.Series, asof=None, ma_days=None) -> bool:
    """大盤濾網:基準指數收盤 >= N 日均線 → True(可持股)。"""
    ma_days = ma_days or config.MARKET_FILTER_MA
    s = bench.loc[:asof] if asof is not None else bench
    s = s.dropna()
    if len(s) < ma_days:
        return True
    return s.iloc[-1] >= s.iloc[-ma_days:].mean()


def target_holdings(prices: pd.DataFrame, bench: pd.Series, asof=None,
                    top_n=None) -> list:
    """回傳本週應持有的股票清單(空清單 = 全現金)。"""
    top_n = top_n or config.TOP_N
    if not market_ok(bench, asof):
        return []
    scores = momentum_scores(prices, asof)
    # 只買動能為正的股票:就算排前五,若本身在下跌也不買
    scores = scores[scores > 0]
    return list(scores.head(top_n).index)


def combined_scores(prices: pd.DataFrame, yoy_table: pd.DataFrame,
                    asof=None, mom_weight=0.5) -> pd.Series:
    """動能 + 月營收年增率的綜合分數(兩者的百分位排名加權平均)。

    白話:一半看「股價最近漲多強」,一半看「公司生意最近成長多快」。
    兩個訊號來源獨立(價格 vs 財報),同時強的股票假象機率較低。
    """
    mom = momentum_scores(prices, asof)
    mom = mom[mom > 0]
    if mom.empty:
        return pd.Series(dtype=float)
    ref_date = asof if asof is not None else prices.index[-1]
    from fundamentals import yoy_asof
    yoy = yoy_asof(yoy_table, ref_date)
    common = mom.index.intersection(yoy.index)
    if len(common) == 0:
        return mom          # 沒有營收資料時退回純動能
    m_rank = mom[common].rank(pct=True)
    f_rank = yoy[common].rank(pct=True)
    combo = mom_weight * m_rank + (1 - mom_weight) * f_rank
    return combo.sort_values(ascending=False)


def combined_targets(prices: pd.DataFrame, bench: pd.Series,
                     yoy_table: pd.DataFrame, asof=None, top_n=None) -> list:
    """動能+營收混合策略的目標持股(同樣受大盤濾網管控)。"""
    top_n = top_n or config.TOP_N
    if not market_ok(bench, asof):
        return []
    combo = combined_scores(prices, yoy_table, asof)
    return list(combo.head(top_n).index)
