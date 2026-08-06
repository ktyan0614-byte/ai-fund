# -*- coding: utf-8 -*-
"""蒙地卡羅對照:主動策略的實際報酬,落在「隨機選股買進持有」分佈的第幾百分位。

用途:把「動能到底有沒有用」壓縮成每週一個數字。
與帳戶二(真實隨機對照組)互補——
  帳戶二是前瞻的、單一次抽籤、不可竄改;
  本模組是回顧的、數千次抽籤、給出完整分佈與百分位。

方法:
  1. 取主動帳戶成立日至今的期間
  2. 從當時可交易的投資範圍中,隨機抽 N 檔、等權、買進持有(僅計買進手續費)
  3. 重複數千次,得到「無選股能力」的報酬分佈
  4. 回報主動帳戶落在此分佈的百分位

百分位 50 = 與丟骰子無異;百分位 5 以下 = 主動流程在扣分;95 以上才是有訊號。
"""
import random

import numpy as np
import pandas as pd

import config

N_PICK = 10          # 每次模擬抽幾檔
N_SIMS = 2000        # 模擬次數
SEED = 42            # 固定種子,任何人可重現


def run(px_all: pd.DataFrame, start_date, actual_return: float,
        n_pick: int = N_PICK, n_sims: int = N_SIMS, seed: int = SEED) -> dict:
    """回傳百分位與分佈統計;資料不足時回傳 None。

    px_all: 日收盤價寬表(index=日期, columns=代號)
    start_date: 主動帳戶成立日(字串或 Timestamp)
    actual_return: 主動帳戶自成立以來的實際報酬(小數,例 -0.198)
    """
    start = pd.Timestamp(start_date)
    win = px_all.loc[px_all.index >= start]
    if len(win) < 2:
        return None

    # 期間內全程有報價的台股(排除中途上市/停止交易者,避免存活偏誤)
    pool = [t for t in config.UNIVERSE
            if t in win.columns and win[t].notna().all()]
    if len(pool) < n_pick + 5:
        return None

    ret = (win.iloc[-1] / win.iloc[0] - 1)[pool]
    buy_cost = max(config.FEE_RATE, config.MIN_FEE / 10000)   # 買進持有:僅計買進費

    rng = random.Random(seed)
    sims = np.array([np.mean([ret[t] for t in rng.sample(pool, n_pick)]) - buy_cost
                     for _ in range(n_sims)])

    return {
        "start": win.index[0].date().isoformat(),
        "end": win.index[-1].date().isoformat(),
        "pool": len(pool),
        "n_pick": n_pick,
        "n_sims": n_sims,
        "seed": seed,
        "actual": actual_return,
        "pct": float((sims < actual_return).mean()),
        "beat_by": float((sims > actual_return).mean()),
        "q": {q: float(np.quantile(sims, q / 100)) for q in (0, 5, 25, 50, 75, 95, 100)},
        "mean": float(sims.mean()),
        "std": float(sims.std()),
    }


def report_lines(mc: dict, label: str = "帳戶一 純動能") -> list:
    """把結果轉成週報用的 markdown 區塊。"""
    if not mc:
        return []
    q = mc["q"]
    lines = [
        "# 蒙地卡羅對照:主動流程落在「隨機選股」的第幾百分位", "",
        f"期間 {mc['start']} ~ {mc['end']}｜從 {mc['pool']} 檔投資範圍隨機抽 "
        f"{mc['n_pick']} 檔、等權、買進持有｜{mc['n_sims']:,} 次模擬（種子 {mc['seed']}）", "",
        "|隨機選股的報酬分佈|報酬|", "|---|---|",
        f"|最差的一次（0 百分位）|{q[0]:+.1%}|",
        f"|5 百分位|{q[5]:+.1%}|",
        f"|**中位數**|**{q[50]:+.1%}**|",
        f"|95 百分位|{q[95]:+.1%}|",
        f"|最好的一次（100 百分位）|{q[100]:+.1%}|",
        "",
        f"**{label} 實際報酬 {mc['actual']:+.1%} → 落在第 "
        f"{mc['pct'] * 100:.1f} 百分位，有 {mc['beat_by'] * 100:.1f}% 的隨機組合贏過它。**",
        "",
        "> 判讀：百分位 50 = 與丟骰子無異；5 以下 = 主動流程在扣分（集中度、"
        "週轉成本、濾網時機皆計入）；95 以上才代表可能有訊號。單期為 n=1，看趨勢不看單次。",
        "",
    ]
    return lines
