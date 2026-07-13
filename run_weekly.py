# -*- coding: utf-8 -*-
"""AI 虛擬投資公司 — 每週執行主程式

流程:
  1. 讀取虛擬帳戶(portfolio/portfolio.json,首次執行自動建立,10 萬元現金)
  2. 抓最新股價,依「動能選股 + 大盤濾網」計算本週目標持股
  3. 以最新收盤價模擬買賣(含手續費、證交稅),更新帳戶
  4. 產出中文週報 reports/YYYY-MM-DD.md(操作內容/決策依據/市場方向/產業近況)
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
import strategy
from data import fetch_prices

BASE = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_PATH = os.path.join(BASE, "portfolio", "portfolio.json")
TRADES_PATH = os.path.join(BASE, "portfolio", "trades.csv")
REPORT_DIR = os.path.join(BASE, "reports")

ALL_TICKERS = list(config.UNIVERSE) + [config.BENCHMARK]


def zh_name(t):
    return config.UNIVERSE.get(t, (config.BENCHMARK_NAME, ""))[0]


def load_portfolio():
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "inception": None,
        "as_of": None,
        "cash": float(config.INITIAL_CASH),
        "positions": {},           # ticker -> {"shares": int, "avg_cost": float}
        "nav_history": [],         # [{"date","nav","bench"}]
    }


def save_portfolio(p):
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)


def append_trades(rows):
    df = pd.DataFrame(rows)
    header = not os.path.exists(TRADES_PATH)
    df.to_csv(TRADES_PATH, mode="a", index=False, header=header,
              encoding="utf-8-sig")


def trade_cost(value, is_sell):
    fee = max(config.MIN_FEE, value * config.FEE_RATE)
    tax = value * config.TAX_RATE if is_sell else 0.0
    return round(fee + tax, 2)


def rebalance(p, px, targets, scores, today):
    """把帳戶調整到目標持股(等權重),回傳交易紀錄清單。"""
    trades = []
    pos = p["positions"]

    def rank_reason(t):
        if t in scores.index:
            r = list(scores.index).index(t) + 1
            return f"動能排名第 {r}(近{config.MOM_LOOKBACK}日 {scores[t]:+.1%})"
        return ""

    # 1. 賣出不在目標內的持股
    for t in list(pos):
        if t not in targets and pos[t]["shares"] > 0:
            n = pos[t]["shares"]
            value = n * px[t]
            cost = trade_cost(value, is_sell=True)
            p["cash"] += value - cost
            pnl = (px[t] - pos[t]["avg_cost"]) * n
            reason = "跌出動能前五,汰弱換強" if targets else "大盤跌破均線濾網,轉為現金避險"
            trades.append({"日期": today, "代號": t, "名稱": zh_name(t),
                           "動作": "賣出", "股數": n, "價格": round(px[t], 2),
                           "金額": round(value, 0), "費用稅": cost,
                           "損益": round(pnl, 0), "原因": reason})
            del pos[t]

    # 2. 買進 / 加碼到等權重
    if targets:
        nav_now = p["cash"] + sum(v["shares"] * px[t] for t, v in pos.items())
        per_stock = nav_now / len(targets)
        for t in targets:
            cur_val = pos.get(t, {}).get("shares", 0) * px[t]
            diff = per_stock - cur_val
            if diff > px[t]:
                affordable = int(p["cash"] / (px[t] * (1 + config.FEE_RATE)))
                buy_n = min(int(diff // px[t]), affordable)
                if buy_n > 0:
                    value = buy_n * px[t]
                    cost = trade_cost(value, is_sell=False)
                    old = pos.get(t, {"shares": 0, "avg_cost": 0.0})
                    new_shares = old["shares"] + buy_n
                    new_cost = (old["shares"] * old["avg_cost"] + value) / new_shares
                    pos[t] = {"shares": new_shares, "avg_cost": round(new_cost, 2)}
                    p["cash"] -= value + cost
                    action = "加碼" if old["shares"] else "買進"
                    trades.append({"日期": today, "代號": t, "名稱": zh_name(t),
                                   "動作": action, "股數": buy_n,
                                   "價格": round(px[t], 2), "金額": round(value, 0),
                                   "費用稅": cost, "損益": "",
                                   "原因": rank_reason(t)})
    p["cash"] = round(p["cash"], 2)
    return trades


def sector_summary(prices):
    """各產業近 1 週 / 1 個月平均報酬。"""
    rows = []
    r1w = prices.iloc[-1] / prices.iloc[-6] - 1 if len(prices) > 6 else None
    r1m = prices.iloc[-1] / prices.iloc[-21] - 1 if len(prices) > 21 else None
    sectors = {}
    for t, (name, sec) in config.UNIVERSE.items():
        sectors.setdefault(sec, []).append(t)
    for sec, ts in sectors.items():
        ts = [t for t in ts if t in prices.columns]
        rows.append({
            "產業": sec,
            "近一週": float(np.mean([r1w[t] for t in ts])) if r1w is not None else np.nan,
            "近一月": float(np.mean([r1m[t] for t in ts])) if r1m is not None else np.nan,
            "代表股": "、".join(zh_name(t) for t in ts),
        })
    return pd.DataFrame(rows).sort_values("近一週", ascending=False)


def market_view(bench):
    """大盤方向的量化描述。"""
    s = bench.dropna()
    last = s.iloc[-1]
    view = {
        "收盤": last,
        "週漲跌": last / s.iloc[-6] - 1 if len(s) > 6 else np.nan,
        "月漲跌": last / s.iloc[-21] - 1 if len(s) > 21 else np.nan,
        "站上20日線": last >= s.iloc[-20:].mean(),
        "站上60日線": last >= s.iloc[-60:].mean(),
        "站上120日線": last >= s.iloc[-120:].mean(),
    }
    n_up = sum(view[k] for k in ["站上20日線", "站上60日線", "站上120日線"])
    view["判讀"] = {3: "多頭趨勢明確", 2: "偏多但短線震盪",
                    1: "趨勢轉弱,保守應對", 0: "空頭趨勢,現金為王"}[n_up]
    return view


def build_report(p, trades, scores, filter_on, mkt, sectors, px, today):
    pos = p["positions"]
    stock_value = sum(v["shares"] * px[t] for t, v in pos.items())
    nav = p["cash"] + stock_value
    lines = []
    lines.append(f"# AI 虛擬投資公司 週報 — {today}")
    lines.append("")

    # 決策卡:給人看的下次開盤行動指示(anti-FOMO)
    lines += ["## 📋 決策卡 — 下個交易日的行動指示", ""]
    if trades:
        lines.append("|順序|動作|股票|股數|參考價(收盤)|掛單上限價|")
        lines.append("|---|---|---|---|---|---|")
        seq = 1
        for tr in [t for t in trades if t["動作"] == "賣出"]:
            lines.append(f"|{seq}|賣出全部|{tr['名稱']}({tr['代號'].replace('.TW','')})"
                         f"|{tr['股數']}|{tr['價格']}|開盤市價賣出即可|")
            seq += 1
        for tr in [t for t in trades if t["動作"] in ("買進", "加碼")]:
            cap = tr["價格"] * 1.02
            lines.append(f"|{seq}|{tr['動作']}|{tr['名稱']}({tr['代號'].replace('.TW','')})"
                         f"|{tr['股數']}|{tr['價格']}|**{cap:,.1f}**(收盤價+2%)|")
            seq += 1
        lines += [
            "",
            "**三條鐵律(寫給未來的自己):**",
            "",
            "1. 買單一律掛**限價**,上限 = 上週收盤價 +2%。開盤跳空漲超過 2% → **放棄,不追**。",
            "2. 買不到不是損失。錯過的股票下週會重新排名,強者恆強自然還會入選。",
            "3. 一週只看盤一次(就是現在)。盤中的任何漲跌都不構成行動理由。",
            "",
        ]
    else:
        lines += ["**本週不需要任何動作。** 持股續抱,不用打開看盤軟體。", ""]

    # 帳戶總覽
    inception_nav = config.INITIAL_CASH
    total_ret = nav / inception_nav - 1
    hist = p["nav_history"]
    week_ret = nav / hist[-1]["nav"] - 1 if hist else 0.0
    bench_ret = ""
    if hist:
        b0 = hist[0]["bench"]
        bench_total = mkt["收盤"] / b0 - 1
        bench_ret = f"|同期 {config.BENCHMARK_NAME}|{bench_total:+.2%}|"
    lines += [
        "## 帳戶總覽",
        "",
        "|項目|數值|",
        "|---|---|",
        f"|總資產(淨值)|{nav:,.0f} 元|",
        f"|現金|{p['cash']:,.0f} 元|",
        f"|持股市值|{stock_value:,.0f} 元|",
        f"|本週損益|{week_ret:+.2%}|",
        f"|成立以來報酬|{total_ret:+.2%}|",
    ]
    if bench_ret:
        lines.append(bench_ret)
    lines.append("")

    # 一、操作內容
    lines += ["## 一、本週操作內容", ""]
    if trades:
        lines.append("|動作|股票|股數|成交價|金額|費用+稅|原因|")
        lines.append("|---|---|---|---|---|---|---|")
        for tr in trades:
            lines.append(f"|{tr['動作']}|{tr['名稱']}({tr['代號'].replace('.TW','')})"
                         f"|{tr['股數']}|{tr['價格']}|{tr['金額']:,.0f}"
                         f"|{tr['費用稅']}|{tr['原因']}|")
    else:
        lines.append("本週無交易:目前持股仍符合策略條件,續抱。")
    lines.append("")

    # 目前持股
    lines += ["### 目前持股", ""]
    if pos:
        lines.append("|股票|股數|成本價|現價|市值|未實現損益|")
        lines.append("|---|---|---|---|---|---|")
        for t, v in pos.items():
            mv = v["shares"] * px[t]
            upnl = (px[t] / v["avg_cost"] - 1)
            lines.append(f"|{zh_name(t)}({t.replace('.TW','')})|{v['shares']}"
                         f"|{v['avg_cost']}|{px[t]:.2f}|{mv:,.0f}|{upnl:+.2%}|")
    else:
        lines.append("目前 100% 現金(避險模式)。")
    lines.append("")

    # 二、決策依據
    lines += [
        "## 二、決策依據",
        "",
        "本公司採用「**動能選股 + 大盤濾網**」策略(經 2020–2026 年回測驗證,"
        "年化報酬約 +59%、優於同期 0050 的 +31%):",
        "",
        f"1. **大盤濾網**:0050 收盤價 {'**站上**' if filter_on else '**跌破**'}"
        f" {config.MARKET_FILTER_MA} 日均線 → "
        f"{'可持股' if filter_on else '全數轉現金避險'}。",
        f"2. **動能排名**:比較 {len(config.UNIVERSE)} 檔大型權值股"
        f"近 {config.MOM_LOOKBACK} 個交易日報酬率,持有前 {config.TOP_N} 名(等權重)。",
        "",
        "本週動能排名前十:",
        "",
        "|排名|股票|產業|近60日報酬|",
        "|---|---|---|---|",
    ]
    for i, (t, sc) in enumerate(scores.head(10).items(), 1):
        star = " ★持有" if t in pos else ""
        lines.append(f"|{i}|{zh_name(t)}({t.replace('.TW','')}){star}"
                     f"|{config.UNIVERSE[t][1]}|{sc:+.1%}|")
    lines.append("")

    # 三、市場方向
    lines += [
        "## 三、近期市場方向",
        "",
        f"以 {config.BENCHMARK_NAME} 觀察整體台股:",
        "",
        "|指標|數值|",
        "|---|---|",
        f"|收盤價|{mkt['收盤']:.2f}|",
        f"|近一週|{mkt['週漲跌']:+.2%}|",
        f"|近一月|{mkt['月漲跌']:+.2%}|",
        f"|20日均線|{'站上 ✅' if mkt['站上20日線'] else '跌破 ❌'}|",
        f"|60日均線|{'站上 ✅' if mkt['站上60日線'] else '跌破 ❌'}|",
        f"|120日均線|{'站上 ✅' if mkt['站上120日線'] else '跌破 ❌'}|",
        "",
        f"**判讀:{mkt['判讀']}**",
        "",
    ]

    # 四、產業近況
    lines += [
        "## 四、各產業近況",
        "",
        "|產業|近一週|近一月|成分股|",
        "|---|---|---|---|",
    ]
    for _, r in sectors.iterrows():
        lines.append(f"|{r['產業']}|{r['近一週']:+.2%}|{r['近一月']:+.2%}|{r['代表股']}|")
    best = sectors.iloc[0]
    worst = sectors.iloc[-1]
    lines += [
        "",
        f"本週最強產業為**{best['產業']}**({best['近一週']:+.2%});"
        f"最弱為**{worst['產業']}**({worst['近一週']:+.2%})。",
        "",
        "---",
        "",
        "> 本報告由 AI 自動產生,為虛擬資金模擬,不構成任何投資建議。",
        "> 看不懂的名詞請查 [GLOSSARY.md](../GLOSSARY.md)(統計系版白話手冊)。",
        "",
    ]
    return "\n".join(lines), nav


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"=== AI 虛擬投資公司 每週執行 {today} ===")

    print("下載最新股價...")
    prices_all = fetch_prices(ALL_TICKERS, start=(pd.Timestamp.now()
                              - pd.Timedelta(days=400)).strftime("%Y-%m-%d"))
    prices_all = prices_all.ffill()
    bench = prices_all[config.BENCHMARK]
    prices = prices_all[list(config.UNIVERSE)]
    px = prices_all.iloc[-1]
    data_date = prices_all.index[-1].strftime("%Y-%m-%d")
    print(f"最新資料日: {data_date}")

    p = load_portfolio()
    if p["inception"] is None:
        p["inception"] = today
        print(f"首次執行:建立虛擬帳戶,初始資金 {config.INITIAL_CASH:,} 元")

    scores = strategy.momentum_scores(prices)
    filter_on = strategy.market_ok(bench)
    targets = strategy.target_holdings(prices, bench)
    targets = [t for t in targets if not np.isnan(px.get(t, np.nan))]
    print(f"大盤濾網: {'通過(可持股)' if filter_on else '未通過(全現金)'}")
    print(f"目標持股: {[zh_name(t) for t in targets] or '無(現金)'}")

    trades = rebalance(p, px, targets, scores, today)
    if trades:
        append_trades(trades)
    print(f"本週交易 {len(trades)} 筆")

    mkt = market_view(bench)
    sectors = sector_summary(prices)
    report, nav = build_report(p, trades, scores, filter_on, mkt, sectors, px, today)

    p["as_of"] = today
    # 同一天重複執行時覆蓋當日紀錄,避免淨值歷史出現重複
    if p["nav_history"] and p["nav_history"][-1]["date"] == today:
        p["nav_history"].pop()
    p["nav_history"].append({"date": today, "nav": round(nav, 2),
                             "bench": round(float(mkt["收盤"]), 2)})
    save_portfolio(p)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"{today}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"週報已產出: {report_path}")
    print(f"目前總資產: {nav:,.0f} 元")


if __name__ == "__main__":
    main()
