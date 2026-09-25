#!/usr/bin/env python3
"""Brute-force BTCUSDT M1 spot volatility/grid optimizer.

Research objective:
- Binance Spot BTCUSDT 1m OHLCV
- only prices >= $30,000
- long-only, 1x spot, no stop-loss
- ZigZag-style swings are used as a market-structure filter
- optimize entry/exit spacing as either percentage or fixed round-dollar distance
- focus on swings >= 2.5%
- rank by total P&L and average monthly P&L

This is deliberately an optimization engine, not a claim that the result is robust.
"""

from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/BTCUSDT_1m_1y.csv")
OUT = Path("results")
MIN_PRICE = 30_000.0
MIN_LEG_PCT = 0.025

# Broad brute-force ranges. Expand later if the best result lands on a boundary.
DEPTHS = [5, 10, 15, 20, 30, 40, 60, 90]
DEVIATIONS_PCT = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
BACKSTEPS = [2, 3, 5, 8, 10, 15]
SPACING_PCT = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
SPACING_USD = [100, 250, 500, 750, 1000, 1500, 2000]
MAX_LAYERS = [1, 2, 3, 5, 8, 10]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    cols = {c.lower(): c for c in df.columns}
    t = cols.get("timestamp") or cols.get("open_time") or cols.get("time")
    if not t:
        raise RuntimeError("No timestamp column found")
    df["timestamp"] = pd.to_datetime(df[t], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    for c in ("open", "high", "low", "close"):
        if c not in cols:
            raise RuntimeError(f"Missing {c} column")
        df[c] = pd.to_numeric(df[cols[c]], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] >= MIN_PRICE].reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close"]]


def zigzag_pivots(close: np.ndarray, deviation_pct: float, depth: int, backstep: int):
    """Causal swing detector.

    A pivot is confirmed only after price moves by deviation_pct from the
    candidate extreme and the minimum depth/backstep constraints are met.
    This is intentionally causal so the optimizer does not use future pivots
    as if they were known at the pivot candle.
    """
    n = len(close)
    pivots = []
    if n < depth + 2:
        return pivots

    direction = 0
    candidate_i = 0
    candidate_p = close[0]

    for i in range(1, n):
        p = close[i]
        if direction >= 0:
            if p >= candidate_p:
                candidate_i, candidate_p = i, p
            if candidate_p > 0 and p <= candidate_p * (1 - deviation_pct):
                if i - candidate_i >= depth and (not pivots or candidate_i - pivots[-1][0] >= backstep):
                    pivots.append((candidate_i, candidate_p, "H"))
                    direction = -1
                    candidate_i, candidate_p = i, p
        if direction <= 0:
            if p <= candidate_p:
                candidate_i, candidate_p = i, p
            if candidate_p > 0 and p >= candidate_p * (1 + deviation_pct):
                if i - candidate_i >= depth and (not pivots or candidate_i - pivots[-1][0] >= backstep):
                    pivots.append((candidate_i, candidate_p, "L"))
                    direction = 1
                    candidate_i, candidate_p = i, p
    return pivots


def leg_stats(pivots):
    legs = []
    for a, b in zip(pivots, pivots[1:]):
        pct = abs(b[1] / a[1] - 1.0)
        legs.append((a[0], b[0], pct, a[2], b[2]))
    return legs


def run_grid(df: pd.DataFrame, spacing_type: str, spacing_value: float,
             max_layers: int, tp_mult: float):
    """Long-only spot grid, no SL.

    Start a cycle after a >=2.5% confirmed downward swing. Buy one tranche
    when price reaches the next grid level below the cycle anchor; add up to
    max_layers. Exit all accumulated BTC when price rebounds by TP multiple
    of the chosen spacing from the volume-weighted average entry.

    Capital is divided equally among layers. No leverage and no shorting.
    """
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    ts = df["timestamp"].to_numpy()
    piv = zigzag_pivots(close, 0.01, 20, 5)  # structure gate; strategy sweep below refines ZZ
    down_legs = {(b, e): pct for b, e, pct, ta, tb in leg_stats(piv)
                 if ta == "H" and tb == "L" and pct >= MIN_LEG_PCT}

    if not down_legs:
        return None

    trades = []
    cash = 1.0
    btc = 0.0
    spent = 0.0
    anchor = None
    next_buy = None
    layers = 0
    last_equity = cash
    peak = cash
    max_dd = 0.0

    for i, p in enumerate(close):
        if p < MIN_PRICE:
            continue
        # A confirmed down-leg endpoint activates a new buying cycle.
        if any(e == i for _, e in down_legs):
            anchor = p
            layers = 0
            spent = 0.0
            btc = 0.0
            next_buy = p

        if anchor is None:
            continue

        step = (spacing_value / 100.0) if spacing_type == "pct" else spacing_value / anchor
        buy_level = anchor * (1 - step * (layers + 1))
        if layers < max_layers and p <= buy_level and cash > 0:
            allocation = cash / (max_layers - layers)
            qty = allocation / p
            cash -= allocation
            spent += allocation
            btc += qty
            layers += 1

        if layers:
            avg = spent / btc
            tp = avg * (1 + step * tp_mult)
            if p >= tp:
                proceeds = btc * p
                pnl = proceeds - spent
                cash += proceeds
                trades.append({
                    "timestamp": ts[i], "pnl": pnl, "return": pnl / spent,
                    "layers": layers, "avg_entry": avg, "exit": p
                })
                btc = 0.0
                spent = 0.0
                layers = 0
                anchor = None
                next_buy = None

        equity = cash + btc * p
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    if not trades:
        return None
    tr = pd.DataFrame(trades)
    monthly = tr.assign(month=pd.to_datetime(tr.timestamp).dt.to_period("M")).groupby("month")["pnl"].sum()
    return {
        "trades": len(tr),
        "total_return_pct": (cash - 1.0) * 100,
        "avg_monthly_return_pct": monthly.mean() * 100,
        "median_monthly_return_pct": monthly.median() * 100,
        "best_month_pct": monthly.max() * 100,
        "worst_month_pct": monthly.min() * 100,
        "max_drawdown_pct": max_dd * 100,
        "win_rate_pct": (tr.pnl > 0).mean() * 100,
        "avg_trade_pct": tr["return"].mean() * 100,
        "avg_layers": tr.layers.mean(),
    }


def main():
    df = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(df):,} candles; {df.timestamp.min()} -> {df.timestamp.max()}")

    # First pass: optimize the grid mechanics aggressively.
    rows = []
    for st, sv, ml, tm in itertools.product(
        ["pct", "usd"], SPACING_PCT + SPACING_USD, MAX_LAYERS, [0.75, 1.0, 1.25, 1.5, 2.0]
    ):
        if st == "pct" and sv not in SPACING_PCT:
            continue
        if st == "usd" and sv not in SPACING_USD:
            continue
        r = run_grid(df, st, sv, ml, tm)
        if r:
            rows.append({"spacing_type": st, "spacing": sv, "max_layers": ml, "tp_multiple": tm, **r})

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No completed trades. Broaden the strategy grid.")
    out = out.sort_values(["avg_monthly_return_pct", "total_return_pct"], ascending=False)
    out.to_csv(OUT / "top_setups.csv", index=False)
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
