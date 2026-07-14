# -*- coding: utf-8 -*-
"""AI 虛擬投資公司 — 每週執行主程式(雙策略 A/B 對照)

兩個各 10 萬元的虛擬帳戶平行操作,做真正的樣本外對照實驗:
  帳戶一「純動能」   :動能 Top5 + 大盤濾網(回測年化 +59%,回檔 -38%)
  帳戶二「動能+營收」 :動能與月營收年增率各半的綜合排名 + 大盤濾網
                      (回測年化 +54%,回檔 -34%,空頭年明顯抗跌)

流程:抓股價與月營收 → 各帳戶計算目標持股 → 模擬買賣(含費稅)→ 產出合併週報
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
REPORT_DIR = os.path.join(BASE, "reports")

ALL_TICKERS = list(config.UNIVERSE) + [config.BENCHMARK]

ACCOUNTS = [
    {"key": "momentum", "name": "帳戶一:純動能",
     "pf": "portfolio.json", "trades": "trades.csv",
     "desc": "近 60 日報酬率前 5 名,等權重"},
    {"key": "hybrid", "name": "帳戶二:動能+營收",
     "pf": "portfolio_hybrid.json", "trades": "trades_hybrid.csv",
     "desc": "動能與月營收年增率各半的綜合排名前 5 名,等權重"},
    {"key": "margin", "name": "帳戶三:動能+毛利",
     "pf": "portfolio_margin.json", "trades": "trades_margin.csv",
     "desc": "動能與毛利率年變化各半的綜合排名前 5 名,等權重"},
    {"key": "us_etf", "name": "帳戶四:美股ETF",
     "pf": "portfolio_us.json", "trades": "trades_us.csv",
     "desc": "SPY 50%/QQQ 30%/VIG 20% 被動持有(台幣計價),偏離目標逾 5 個百分點才再平衡"},
]


def zh_name(t):
    if t in config.UNIVERSE:
        return config.UNIVERSE[t][0]
    if t in config.US_ETF_PORTFOLIO:
        return config.US_ETF_PORTFOLIO[t][0]
    return config.BENCHMARK_NAME


def pf_path(acct):
    return os.path.join(BASE, "portfolio", acct["pf"])


def trades_path(acct):
    return os.path.join(BASE, "portfolio", acct["trades"])


def load_portfolio(acct):
    path = pf_path(acct)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"inception": None, "as_of": None,
            "cash": float(config.INITIAL_CASH),
            "positions": {}, "nav_history": []}


def save_portfolio(acct, p):
    with open(pf_path(acct), "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)


def append_trades(acct, rows):
    path = trades_path(acct)
    pd.DataFrame(rows).to_csv(path, mode="a", index=False,
                              header=not os.path.exists(path),
                              encoding="utf-8-sig")


def todays_trades(acct, today):
    """讀出今天全部的交易(同日重跑時報告才不會洗掉先前紀錄)。"""
    path = trades_path(acct)
    if not os.path.exists(path):
        return []
    tdf = pd.read_csv(path, encoding="utf-8-sig", dtype={"日期": str})
    return tdf[tdf["日期"] == today].fillna("").to_dict("records")


def trade_cost(value, is_sell, ticker=""):
    if ticker and not ticker.endswith(".TW"):     # 海外 ETF:低費率、無證交稅
        return round(value * config.US_FEE_RATE, 2)
    fee = max(config.MIN_FEE, value * config.FEE_RATE)
    tax = value * config.TAX_RATE if is_sell else 0.0
    return round(fee + tax, 2)


def rebalance(p, px, targets, reason_fn, today, weights=None):
    """把帳戶調整到目標持股。weights=None 時等權重;台股整數股,海外 ETF 可碎股。"""
    trades = []
    pos = p["positions"]

    def w(t):
        return weights[t] if weights else 1 / len(targets)

    # 1. 賣出不在目標內的持股
    for t in list(pos):
        if t not in targets and pos[t]["shares"] > 0 and not np.isnan(px[t]):
            n = pos[t]["shares"]
            value = n * px[t]
            cost = trade_cost(value, is_sell=True, ticker=t)
            p["cash"] += value - cost
            pnl = (px[t] - pos[t]["avg_cost"]) * n
            reason = "跌出排名前五,汰弱換強" if targets else "大盤跌破均線濾網,轉為現金避險"
            trades.append({"日期": today, "代號": t, "名稱": zh_name(t),
                           "動作": "賣出", "股數": n, "價格": round(px[t], 2),
                           "金額": round(value, 0), "費用稅": cost,
                           "損益": round(pnl, 0), "原因": reason})
            del pos[t]

    # 2. 買進 / 加碼 / 減碼到目標配置
    if targets:
        nav_now = p["cash"] + sum(v["shares"] * px[t] for t, v in pos.items())
        for t in targets:
            frac = not t.endswith(".TW")          # 海外 ETF 允許碎股
            target_val = nav_now * w(t)
            cur_val = pos.get(t, {}).get("shares", 0) * px[t]
            diff = target_val - cur_val
            min_trade = px[t] * (0.001 if frac else 1)
            if diff > min_trade:                  # 買進/加碼
                if frac:
                    buy_n = round(min(diff, p["cash"] / (1 + config.US_FEE_RATE)) / px[t], 4)
                else:
                    affordable = int(p["cash"] / (px[t] * (1 + config.FEE_RATE)))
                    buy_n = min(int(diff // px[t]), affordable)
                if buy_n > 0:
                    value = buy_n * px[t]
                    cost = trade_cost(value, is_sell=False, ticker=t)
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
                                   "原因": reason_fn(t)})
            elif diff < -min_trade and t in pos:  # 減碼(再平衡賣出一部分)
                sell_n = (round((-diff) / px[t], 4) if frac
                          else min(int((-diff) // px[t]), pos[t]["shares"]))
                sell_n = min(sell_n, pos[t]["shares"])
                if sell_n > 0:
                    value = sell_n * px[t]
                    cost = trade_cost(value, is_sell=True, ticker=t)
                    p["cash"] += value - cost
                    pos[t]["shares"] = round(pos[t]["shares"] - sell_n, 4)
                    trades.append({"日期": today, "代號": t, "名稱": zh_name(t),
                                   "動作": "減碼", "股數": sell_n,
                                   "價格": round(px[t], 2), "金額": round(value, 0),
                                   "費用稅": cost, "損益": "",
                                   "原因": reason_fn(t)})
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


def account_section(acct, p, trades, ranking_lines, px, mkt):
    """單一帳戶的報告區塊(決策卡/總覽/操作/持股/排名),回傳 (lines, nav)。"""
    pos = p["positions"]
    stock_value = sum(v["shares"] * px[t] for t, v in pos.items())
    nav = p["cash"] + stock_value
    suffix = "" if acct["key"] == "us_etf" else " + 大盤濾網"
    lines = [f"# {acct['name']}", "", f"策略:{acct['desc']}{suffix}", ""]

    # 決策卡
    lines += ["### 📋 決策卡 — 下個交易日的行動指示", ""]
    if trades:
        lines.append("|順序|動作|股票|股數|參考價(收盤)|掛單上限價|")
        lines.append("|---|---|---|---|---|---|")
        seq = 1
        for tr in [t for t in trades if t["動作"] == "賣出"]:
            lines.append(f"|{seq}|賣出全部|{tr['名稱']}({str(tr['代號']).replace('.TW','')})"
                         f"|{tr['股數']}|{tr['價格']}|開盤市價賣出即可|")
            seq += 1
        for tr in [t for t in trades if t["動作"] in ("買進", "加碼")]:
            cap = float(tr["價格"]) * 1.02
            lines.append(f"|{seq}|{tr['動作']}|{tr['名稱']}({str(tr['代號']).replace('.TW','')})"
                         f"|{tr['股數']}|{tr['價格']}|**{cap:,.1f}**(收盤價+2%)|")
            seq += 1
        lines.append("")
    else:
        lines += ["**本週不需要任何動作。** 持股續抱。", ""]

    # 帳戶總覽
    hist = p["nav_history"]
    week_ret = nav / hist[-1]["nav"] - 1 if hist else 0.0
    total_ret = nav / config.INITIAL_CASH - 1
    lines += ["### 帳戶總覽", "", "|項目|數值|", "|---|---|",
              f"|總資產(淨值)|{nav:,.0f} 元|",
              f"|現金|{p['cash']:,.0f} 元|",
              f"|持股市值|{stock_value:,.0f} 元|",
              f"|本週損益|{week_ret:+.2%}|",
              f"|成立以來報酬|{total_ret:+.2%}|"]
    if hist:
        bench_total = mkt["收盤"] / hist[0]["bench"] - 1
        lines.append(f"|同期 {config.BENCHMARK_NAME}|{bench_total:+.2%}|")
    lines.append("")

    # 本週操作
    lines += ["### 本週操作內容", ""]
    if trades:
        lines.append("|動作|股票|股數|成交價|金額|費用+稅|原因|")
        lines.append("|---|---|---|---|---|---|---|")
        for tr in trades:
            lines.append(f"|{tr['動作']}|{tr['名稱']}({str(tr['代號']).replace('.TW','')})"
                         f"|{tr['股數']}|{tr['價格']}|{float(tr['金額']):,.0f}"
                         f"|{tr['費用稅']}|{tr['原因']}|")
    else:
        lines.append("本週無交易:目前持股仍符合策略條件,續抱。")
    lines.append("")

    # 持股
    lines += ["### 目前持股", ""]
    if pos:
        lines.append("|股票|股數|成本價|現價|市值|未實現損益|")
        lines.append("|---|---|---|---|---|---|")
        for t, v in pos.items():
            mv = v["shares"] * px[t]
            upnl = px[t] / v["avg_cost"] - 1
            lines.append(f"|{zh_name(t)}({t.replace('.TW','')})|{v['shares']}"
                         f"|{v['avg_cost']}|{px[t]:.2f}|{mv:,.0f}|{upnl:+.2%}|")
    else:
        lines.append("目前 100% 現金(避險模式)。")
    lines.append("")

    # 排名依據
    lines += ["### 本週排名(決策依據)", ""] + ranking_lines + [""]
    return lines, nav


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
    print(f"最新資料日: {prices_all.index[-1].strftime('%Y-%m-%d')}")

    print("下載月營收資料...")
    try:
        from fundamentals import fetch_month_revenue, revenue_yoy_table
        rev = fetch_month_revenue(list(config.UNIVERSE))
        yoy_table = revenue_yoy_table(rev)
        yoy_now = yoy_table.ffill().iloc[-1]
    except Exception as e:                      # API 掛掉時退回純動能,不中斷
        print(f"警告:月營收下載失敗({e}),帳戶二本週退回純動能訊號")
        yoy_table = pd.DataFrame()
        yoy_now = pd.Series(dtype=float)

    print("下載季報資料(毛利率)...")
    try:
        from fundamentals import fetch_financials, margin_trend_table
        fin = fetch_financials(list(config.UNIVERSE), start="2023-01-01")
        gm_table = margin_trend_table(fin)
        gm_now = gm_table.ffill().iloc[-1]
    except Exception as e:
        print(f"警告:季報下載失敗({e}),帳戶三本週退回純動能訊號")
        gm_table = pd.DataFrame()
        gm_now = pd.Series(dtype=float)

    print("下載美股 ETF 資料(台幣計價)...")
    try:
        import yfinance as yf
        us_raw = yf.download(list(config.US_ETF_PORTFOLIO) + ["TWD=X"],
                             start=(pd.Timestamp.now()
                                    - pd.Timedelta(days=430)).strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)["Close"]
        fx = us_raw["TWD=X"]
        fx[fx.pct_change().abs() > 0.15] = np.nan     # 匯率壞點防護
        us_twd = (us_raw[list(config.US_ETF_PORTFOLIO)]
                  .mul(fx.ffill(), axis=0).dropna(how="all").ffill())
    except Exception as e:
        print(f"警告:美股資料下載失敗({e}),帳戶四本週跳過")
        us_twd = pd.DataFrame()
    px_all = pd.concat([px, us_twd.iloc[-1]]) if not us_twd.empty else px

    filter_on = strategy.market_ok(bench)
    mom_scores = strategy.momentum_scores(prices)
    print(f"大盤濾網: {'通過(可持股)' if filter_on else '未通過(全現金)'}")

    # --- 各帳戶決策 ---
    mkt = market_view(bench)
    sectors = sector_summary(prices)
    all_lines = [f"# AI 虛擬投資公司 週報 — {today}", "",
                 f"大盤濾網:{'✅ 通過,可持股' if filter_on else '❌ 未通過,全數轉現金'}"
                 f"(0050 vs {config.MARKET_FILTER_MA} 日均線)", "", "---", ""]
    navs = {}

    for acct in ACCOUNTS:
        p = load_portfolio(acct)
        if p["inception"] is None:
            p["inception"] = today
            print(f"{acct['name']}:建立虛擬帳戶,初始資金 {config.INITIAL_CASH:,} 元")

        hold, weights_map = False, None
        if acct["key"] == "us_etf":
            if us_twd.empty:
                print(f"{acct['name']}:美股資料缺,本週跳過")
                continue
            weights_map = {t: w for t, (_, w) in config.US_ETF_PORTFOLIO.items()}
            pos_now = p["positions"]
            nav_now = p["cash"] + sum(v["shares"] * px_all[t]
                                      for t, v in pos_now.items())
            cur_w = {t: pos_now.get(t, {}).get("shares", 0) * px_all[t] / nav_now
                     for t in weights_map}
            is_new = not pos_now
            need = is_new or any(abs(cur_w[t] - weights_map[t])
                                 > config.US_REBALANCE_BAND for t in weights_map)
            targets = list(weights_map)
            hold = not need

            def reason_fn(t, wm=weights_map, new=is_new):
                return (f"美股被動配置至目標 {wm[t]:.0%}"
                        + ("(初始建倉)" if new else "(偏離逾5個百分點,再平衡)"))

            r1y = (us_twd.iloc[-1] / us_twd.iloc[-253] - 1) if len(us_twd) > 253 else None
            ranking = ["|ETF|目標配置|目前配置|近一年報酬(台幣)|", "|---|---|---|---|"]
            for t, (name, tw) in config.US_ETF_PORTFOLIO.items():
                r1 = (f"{r1y[t]:+.1%}" if r1y is not None
                      and not np.isnan(r1y.get(t, np.nan)) else "—")
                ranking.append(f"|{name}({t})|{tw:.0%}|{cur_w.get(t, 0):.0%}|{r1}|")
            ranking += ["", f"再平衡規則:任一 ETF 偏離目標配置逾 "
                        f"{config.US_REBALANCE_BAND:.0%} 才調整;"
                        f"本週{'執行再平衡' if need else '無需動作,續抱'}。"
                        f"此部門不受台股大盤濾網管控(被動長期持有)。"]
        elif acct["key"] == "momentum":
            targets = strategy.target_holdings(prices, bench)
            scores = mom_scores

            def reason_fn(t, s=scores):
                r = list(s.index).index(t) + 1
                return f"動能排名第 {r}(近{config.MOM_LOOKBACK}日 {s[t]:+.1%})"

            ranking = ["|排名|股票|產業|近60日報酬|", "|---|---|---|---|"]
            for i, (t, sc) in enumerate(scores.head(10).items(), 1):
                star = " ★持有" if t in p["positions"] or t in targets else ""
                ranking.append(f"|{i}|{zh_name(t)}({t.replace('.TW','')}){star}"
                               f"|{config.UNIVERSE[t][1]}|{sc:+.1%}|")
        else:
            table, now_vals, label = (
                (yoy_table, yoy_now, "營收年增率(近3月均)")
                if acct["key"] == "hybrid" else
                (gm_table, gm_now, "毛利率年變化"))
            targets = (strategy.combined_targets(prices, bench, table)
                       if not table.empty and filter_on else
                       (strategy.target_holdings(prices, bench) if filter_on else []))
            scores = strategy.combined_scores(prices, table) \
                if not table.empty else mom_scores

            def reason_fn(t, s=scores, m=mom_scores, y=now_vals, lb=label):
                r = list(s.index).index(t) + 1
                return (f"綜合排名第 {r}(動能 {m.get(t, float('nan')):+.1%},"
                        f"{lb} {y.get(t, float('nan')):+.1%})")

            ranking = [f"|排名|股票|產業|近60日報酬|{label}|",
                       "|---|---|---|---|---|"]
            for i, t in enumerate(scores.head(10).index, 1):
                star = " ★持有" if t in p["positions"] or t in targets else ""
                ranking.append(f"|{i}|{zh_name(t)}({t.replace('.TW','')}){star}"
                               f"|{config.UNIVERSE[t][1]}"
                               f"|{mom_scores.get(t, float('nan')):+.1%}"
                               f"|{now_vals.get(t, float('nan')):+.1%}|")

        targets = [t for t in targets if not np.isnan(px_all.get(t, np.nan))]
        print(f"{acct['name']} 目標: {[zh_name(t) for t in targets] or '無(現金)'}"
              + ("(續抱,無需動作)" if hold else ""))

        new_trades = [] if hold else rebalance(p, px_all, targets, reason_fn,
                                               today, weights=weights_map)
        if new_trades:
            append_trades(acct, new_trades)
        trades = todays_trades(acct, today)

        section, nav = account_section(acct, p, trades, ranking, px_all, mkt)
        all_lines += section + ["---", ""]
        navs[acct["name"]] = nav

        p["as_of"] = today
        if p["nav_history"] and p["nav_history"][-1]["date"] == today:
            p["nav_history"].pop()
        p["nav_history"].append({"date": today, "nav": round(nav, 2),
                                 "bench": round(float(mkt["收盤"]), 2)})
        save_portfolio(acct, p)

    # --- 共用區塊:市場方向 / 產業近況 ---
    all_lines += [
        "# 近期市場方向", "",
        f"以 {config.BENCHMARK_NAME} 觀察整體台股:", "",
        "|指標|數值|", "|---|---|",
        f"|收盤價|{mkt['收盤']:.2f}|",
        f"|近一週|{mkt['週漲跌']:+.2%}|",
        f"|近一月|{mkt['月漲跌']:+.2%}|",
        f"|20日均線|{'站上 ✅' if mkt['站上20日線'] else '跌破 ❌'}|",
        f"|60日均線|{'站上 ✅' if mkt['站上60日線'] else '跌破 ❌'}|",
        f"|120日均線|{'站上 ✅' if mkt['站上120日線'] else '跌破 ❌'}|",
        "", f"**判讀:{mkt['判讀']}**", "",
        "# 各產業近況", "",
        "|產業|近一週|近一月|成分股|", "|---|---|---|---|",
    ]
    for _, r in sectors.iterrows():
        all_lines.append(f"|{r['產業']}|{r['近一週']:+.2%}|{r['近一月']:+.2%}|{r['代表股']}|")
    best, worst = sectors.iloc[0], sectors.iloc[-1]
    all_lines += [
        "",
        f"本週最強產業為**{best['產業']}**({best['近一週']:+.2%});"
        f"最弱為**{worst['產業']}**({worst['近一週']:+.2%})。",
        "", "**三條鐵律:** 買單掛限價(上限=收盤+2%),超過不追|"
        "買不到不是損失,下週重新排名|一週只看盤一次",
        "", "---", "",
        "> 本報告由 AI 自動產生,為虛擬資金模擬,不構成任何投資建議。",
        "> 看不懂的名詞請查 [GLOSSARY.md](../GLOSSARY.md)(統計系版白話手冊)。", "",
    ]

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"{today}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print(f"週報已產出: {report_path}")
    for name, nav in navs.items():
        print(f"{name} 總資產: {nav:,.0f} 元")


if __name__ == "__main__":
    main()
