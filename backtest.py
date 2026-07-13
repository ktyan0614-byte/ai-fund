# -*- coding: utf-8 -*-
"""回測:比較幾種策略在歷史資料上的表現,決定虛擬投資公司採用哪一套。

策略候選:
  A. 買進並持有 0050(比較基準,代表「什麼都不做」)
  B. 動能選股 Top5(不加濾網)
  C. 動能選股 Top5 + 大盤 120 日均線濾網(跌破全轉現金)
  D. 0050 均線策略(20 日均線 > 60 日均線才持有,否則現金)

所有策略都以每週一次的頻率決策,並計入台股實際手續費與證交稅。
"""
import numpy as np
import pandas as pd

import config
import strategy
from data import fetch_prices


def trade_cost(value, is_sell):
    fee = max(config.MIN_FEE, value * config.FEE_RATE)
    tax = value * config.TAX_RATE if is_sell else 0.0
    return fee + tax


def simulate(prices: pd.DataFrame, bench: pd.Series, decide_fn,
             initial_cash=config.INITIAL_CASH, warmup=130,
             exec_delay=0, buy_px: pd.DataFrame = None,
             sell_px: pd.DataFrame = None):
    """每週(週五收盤)依 decide_fn 給的目標清單調整持股,回傳每日淨值序列。

    decide_fn(asof) -> list of tickers(等權重)或 [](全現金)
    exec_delay=0: 決策當天以收盤價成交(最樂觀)
    exec_delay=1: 決策後「下一個交易日」才成交,成交價用 buy_px / sell_px
                  (例如下週一開盤價,或最悲觀的買最高/賣最低)
    """
    buy_px = prices if buy_px is None else buy_px
    sell_px = prices if sell_px is None else sell_px
    dates = prices.index
    # 每週最後一個交易日做決策
    week_ends = pd.Series(dates, index=dates).resample("W-FRI").last().dropna()
    week_ends = [d for d in week_ends if dates.get_loc(d) >= warmup]
    if not week_ends:
        return pd.Series(dtype=float)

    cash = float(initial_cash)
    shares = {}          # ticker -> 股數(整數,零股)
    nav_records = {}
    pending = None       # exec_delay=1 時,等待下個交易日執行的目標清單

    def execute(targets, pb, ps):
        nonlocal cash
        targets = [t for t in targets if not np.isnan(pb.get(t, np.nan))]
        # 先賣出不在目標內的
        for t in list(shares):
            if t not in targets and shares[t] > 0 and not np.isnan(ps[t]):
                value = shares[t] * ps[t]
                cash += value - trade_cost(value, is_sell=True)
                shares[t] = 0
        # 再買進 / 調整到等權重
        if targets:
            nav_now = cash + sum(n * pb[t] for t, n in shares.items() if n > 0)
            per_stock = nav_now / len(targets)
            for t in targets:
                cur_val = shares.get(t, 0) * pb[t]
                diff = per_stock - cur_val
                if diff > pb[t]:          # 需要加碼
                    # 預留手續費,避免現金不足買不成
                    affordable = int(cash / (pb[t] * (1 + config.FEE_RATE)))
                    buy_n = min(int(diff // pb[t]), affordable)
                    if buy_n > 0:
                        cost = buy_n * pb[t]
                        shares[t] = shares.get(t, 0) + buy_n
                        cash -= cost + trade_cost(cost, is_sell=False)
                elif diff < -pb[t]:       # 需要減碼
                    sell_n = min(int((-diff) // pb[t]), shares.get(t, 0))
                    if sell_n > 0 and not np.isnan(ps[t]):
                        value = sell_n * ps[t]
                        cash += value - trade_cost(value, is_sell=True)
                        shares[t] -= sell_n

    rebalance_set = set(week_ends)
    for d in dates[dates.get_loc(week_ends[0]):]:
        px = prices.loc[d]
        if pending is not None:                       # 延遲成交:今天執行上次決策
            execute(pending, buy_px.loc[d], sell_px.loc[d])
            pending = None
        if d in rebalance_set:
            targets = decide_fn(d)
            if exec_delay == 0:
                execute(targets, px, px)
            else:
                pending = targets
        nav = cash + sum(n * px[t] for t, n in shares.items()
                         if n > 0 and not np.isnan(px[t]))
        nav_records[d] = nav
    return pd.Series(nav_records, name="nav")


def stats(nav: pd.Series) -> dict:
    ret = nav.iloc[-1] / nav.iloc[0] - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    daily = nav.pct_change(fill_method=None).dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    dd = (nav / nav.cummax() - 1).min()
    return {"總報酬": f"{ret:+.1%}", "年化報酬": f"{cagr:+.1%}",
            "最大回檔": f"{dd:.1%}", "夏普值": f"{sharpe:.2f}"}


def main():
    start = "2020-01-01"
    tickers = list(config.UNIVERSE) + [config.BENCHMARK]
    print("下載歷史股價中...")
    from data import fetch_fields
    ohlc = fetch_fields(tickers, start=start, cache_prefix="backtest")
    prices_all = ohlc["Close"].ffill()   # 個股偶有缺價日,以前一日收盤補
    open_all = ohlc["Open"].ffill()
    high_all = ohlc["High"].ffill()
    low_all = ohlc["Low"].ffill()
    bench = prices_all[config.BENCHMARK]
    prices = prices_all[list(config.UNIVERSE)]
    print(f"資料區間: {prices.index[0].date()} ~ {prices.index[-1].date()}, "
          f"{len(prices)} 個交易日, {prices.shape[1]} 檔股票")

    def decide_A(asof):   # 買進持有 0050
        return [config.BENCHMARK]

    def decide_B(asof):   # 動能 Top5, 無濾網
        scores = strategy.momentum_scores(prices, asof)
        scores = scores[scores > 0]
        return list(scores.head(config.TOP_N).index)

    def decide_C(asof):   # 動能 Top5 + 大盤濾網
        return strategy.target_holdings(prices, bench, asof)

    def decide_D(asof):   # 0050 均線 20/60
        s = bench.loc[:asof].dropna()
        if len(s) < 60:
            return []
        return [config.BENCHMARK] if s.iloc[-20:].mean() > s.iloc[-60:].mean() else []

    navs = {}
    for name, fn in [("A_持有0050", decide_A), ("B_動能Top5", decide_B),
                     ("C_動能+濾網", decide_C), ("D_0050均線", decide_D)]:
        print(f"回測 {name} ...")
        navs[name] = simulate(prices_all, bench, fn)

    # 成交價敏感度測試:同一套 C 策略,換成更貼近實際的成交假設
    print("回測 C_週一開盤成交(週五收盤決策,下個交易日開盤價成交)...")
    navs["C_週一開盤成交"] = simulate(prices_all, bench, decide_C,
                                exec_delay=1, buy_px=open_all, sell_px=open_all)
    print("回測 C_最差成交(買在隔日最高、賣在隔日最低,模擬FOMO追高)...")
    navs["C_最差成交"] = simulate(prices_all, bench, decide_C,
                              exec_delay=1, buy_px=high_all, sell_px=low_all)

    result = pd.DataFrame(navs).dropna()
    result.to_csv("outputs/backtest_navs.csv", encoding="utf-8-sig")

    print("\n=== 回測結果(初始資金 10 萬元)===")
    rows = []
    for name, nav in navs.items():
        s = stats(nav)
        s["策略"] = name
        s["期末資產"] = f"{nav.iloc[-1]:,.0f}"
        rows.append(s)
    table = pd.DataFrame(rows)[["策略", "期末資產", "總報酬", "年化報酬", "最大回檔", "夏普值"]]
    print(table.to_string(index=False))
    table.to_csv("outputs/backtest_summary.csv", index=False, encoding="utf-8-sig")

    # 分年度報酬
    print("\n=== 各年度報酬 ===")
    yearly = result.resample("YE").last().pct_change(fill_method=None)
    first_year = result.resample("YE").last().iloc[0] / result.iloc[0] - 1
    yearly.iloc[0] = first_year
    yearly.index = yearly.index.year
    print(yearly.map(lambda x: f"{x:+.1%}").to_string())
    yearly.to_csv("outputs/backtest_yearly.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
