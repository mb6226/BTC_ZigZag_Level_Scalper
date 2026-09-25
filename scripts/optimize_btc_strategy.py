#!/usr/bin/env python3
"""BTCUSDT M1 spot volatility/grid optimizer.

Research objective:
- Binance Spot BTCUSDT 1m OHLCV
- only prices >= $30,000
- long-only, 1x spot, no stop-loss
- causal percentage ZigZag market-structure filter
- optimize percentage or fixed-dollar grid spacing
- focus on confirmed downward swings >= 2.5%
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

DEPTHS = [5, 10, 15, 20, 30, 40, 60, 90]
DEVIATIONS_PCT = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
BACKSTEPS = [2, 3, 5, 8, 10, 15]
SPACING_PCT = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
SPACING_USD = [100, 250, 500, 750, 1000, 1500, 2000]
MAX_LAYERS = [1, 2, 3, 5, 8, 10]
TP_MULTIPLES = [0.75, 1.0, 1.25, 1.5, 2.0]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    cols = {c.lower(): c for c in df.columns}
    t = cols.get("timestamp") or cols.get("open_time") or cols.get("time")
    if not t:
        raise RuntimeError("No timestamp column found")

    raw_ts = pd.to_numeric(df[t], errors="coerce")
    sample = raw_ts.dropna().iloc[0]
    if sample >= 1e17:
        unit = "ns"
    elif sample >= 1e14:
        unit = "us"
    elif sample >= 1e11:
        unit = "ms"
    else:
        unit = "s"

    df["timestamp"] = pd.to_datetime(raw_ts, unit=unit, utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    for c in ("open", "high", "low", "close"):
        if c not in cols:
            raise RuntimeError(f"Missing {c} column")
        df[c] = pd.to_numeric(df[cols[c]], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] >= MIN_PRICE].reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close"]]


def zigzag_pivots(close: np.ndarray, deviation_pct: float, depth: int, backstep: int):
    """Causal percentage ZigZag.

    Important: the initial candidate is NOT overwritten before testing the
    reversal. The previous implementation did exactly that, making
    p / candidate_p - 1 equal to zero during bootstrap, so direction could
    never leave state 0 and no pivots were ever produced.
    """
    n = len(close)
    if n < depth + 2:
        return []

    pivots = []
    direction = 1  # start by seeking a high
    candidate_i = 0
    candidate_p = float(close[0])

    for i in range(1, n):
        p = float(close[i])

        if direction == 1:
            if p > candidate_p:
                candidate_i, candidate_p = i, p
                continue

            reversal = 1.0 - p / candidate_p
            if (
                reversal >= deviation_pct
                and i - candidate_i >= depth
                and (not pivots or candidate_i - pivots[-1][0] >= backstep)
            ):
                pivots.append((candidate_i, candidate_p, "H"))
                direction = -1
                candidate_i, candidate_p = i, p
            continue

        if p < candidate_p:
            candidate_i, candidate_p = i, p
            continue

        reversal = p / candidate_p - 1.0
        if (
            reversal >= deviation_pct
            and i - candidate_i >= depth
            and (not pivots or candidate_i - pivots[-1][0] >= backstep)
        ):
            pivots.append((candidate_i, candidate_p, "L"))
            direction = 1
            candidate_i, candidate_p = i, p

    return pivots


def leg_stats(pivots):
    return [
        (a[0], b[0], abs(b[1] / a[1] - 1.0), a[2], b[2])
        for a, b in zip(pivots, pivots[1:])
    ]


class RangeTree:
    """Segment tree for first close crossing a threshold after an index."""

    def __init__(self, values):
        n = len(values)
        size = 1
        while size < n:
            size *= 2
        self.n = n
        self.size = size
        self.minv = np.full(2 * size, np.inf, dtype=np.float64)
        self.maxv = np.full(2 * size, -np.inf, dtype=np.float64)
        self.minv[size:size + n] = values
        self.maxv[size:size + n] = values
        for i in range(size - 1, 0, -1):
            self.minv[i] = min(self.minv[2*i], self.minv[2*i+1])
            self.maxv[i] = max(self.maxv[2*i], self.maxv[2*i+1])

    def first_le(self, start, threshold):
        return self._first(start, threshold, True)

    def first_ge(self, start, threshold):
        return self._first(start, threshold, False)

    def _first(self, start, threshold, le):
        if start >= self.n:
            return None

        def walk(node, left, right):
            if right <= start:
                return None
            extreme = self.minv[node] if le else self.maxv[node]
            if (le and extreme > threshold) or ((not le) and extreme < threshold):
                return None
            if right - left == 1:
                return left if left < self.n else None
            mid = (left + right) // 2
            hit = walk(node * 2, left, mid)
            return hit if hit is not None else walk(node * 2 + 1, mid, right)

        return walk(1, 0, self.size)


def run_grid(df, pivots, spacing_type, spacing_value, max_layers, tp_mult, range_tree):
    """Event-driven backtest; avoids scanning all 500k candles per setup."""
    close = df["close"].to_numpy()
    ts = df["timestamp"].to_numpy()

    down_ends = [
        e for _, e, pct, ta, tb in leg_stats(pivots)
        if ta == "H" and tb == "L" and pct >= MIN_LEG_PCT
    ]
    if not down_ends:
        return None

    trades = []
    cash, btc, spent = 1.0, 0.0, 0.0
    layers = 0
    peak, max_dd = cash, 0.0

    for anchor_i in down_ends:
        if layers or close[anchor_i] < MIN_PRICE:
            continue

        anchor = float(close[anchor_i])
        allocation = cash / max_layers
        btc = allocation / anchor
        cash -= allocation
        spent = allocation
        layers = 1
        current_i = anchor_i

        while layers:
            step = spacing_value / 100.0 if spacing_type == "pct" else spacing_value / anchor
            buy_level = anchor * (1.0 - step * (layers + 1))
            avg = spent / btc
            tp = avg * (1.0 + step * tp_mult)

            buy_i = range_tree.first_le(current_i + 1, buy_level) if layers < max_layers else None
            tp_i = range_tree.first_ge(current_i + 1, tp)

            if buy_i is None and tp_i is None:
                equity = cash + btc * float(close[-1])
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
                break

            if buy_i is not None and (tp_i is None or buy_i <= tp_i):
                p = float(close[buy_i])
                allocation = cash / (max_layers - layers)
                btc += allocation / p
                cash -= allocation
                spent += allocation
                layers += 1
                current_i = buy_i
                equity = cash + btc * p
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
                continue

            p = float(close[tp_i])
            proceeds = btc * p
            pnl = proceeds - spent
            cash += proceeds
            trades.append({
                "timestamp": ts[tp_i],
                "pnl": pnl,
                "return": pnl / spent,
                "layers": layers,
                "avg_entry": spent / btc,
                "exit": p,
            })
            btc, spent, layers = 0.0, 0.0, 0
            current_i = tp_i

    if not trades:
        return None

    tr = pd.DataFrame(trades)
    month = pd.to_datetime(tr["timestamp"], utc=True).dt.tz_localize(None).dt.to_period("M")
    monthly = tr.assign(month=month).groupby("month")["pnl"].sum()

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
    print(
        f"Loaded {len(df):,} candles; "
        f"{df.timestamp.min()} -> {df.timestamp.max()}"
    )

    close = df["close"].to_numpy()
    range_tree = RangeTree(close)
    zz_rows = []

    for depth, dev, backstep in itertools.product(
        DEPTHS, DEVIATIONS_PCT, BACKSTEPS
    ):
        piv = zigzag_pivots(close, dev / 100.0, depth, backstep)
        legs = leg_stats(piv)
        down = [
            x for x in legs
            if x[3] == "H" and x[4] == "L" and x[2] >= MIN_LEG_PCT
        ]
        zz_rows.append({
            "depth": depth,
            "deviation_pct": dev,
            "backstep": backstep,
            "down_legs": len(down),
            "median_leg_pct": (
                float(np.median([x[2] for x in down])) if down else 0.0
            ),
            "all_legs": len(legs),
        })

    zz_df = pd.DataFrame(zz_rows).sort_values(
        ["down_legs", "median_leg_pct", "all_legs"],
        ascending=False,
    ).reset_index(drop=True)

    print(
        "ZigZag sweep: max >=2.5% down legs =",
        int(zz_df.down_legs.max()),
    )
    print(zz_df.head(10).to_string(index=False))

    if zz_df.down_legs.max() == 0:
        raise RuntimeError(
            "ZigZag detector found no >=2.5% downward legs; inspect detector/grid."
        )

    top_zz = zz_df.head(20)
    rows = []

    for _, z in top_zz.iterrows():
        depth = int(z["depth"])
        dev = float(z["deviation_pct"])
        backstep = int(z["backstep"])
        piv = zigzag_pivots(close, dev / 100.0, depth, backstep)

        for st, spacings in [("pct", SPACING_PCT), ("usd", SPACING_USD)]:
            for sv, ml, tm in itertools.product(spacings, MAX_LAYERS, TP_MULTIPLES):
                r = run_grid(df, piv, st, sv, ml, tm, range_tree)
            if r:
                rows.append({
                    "depth": depth,
                    "deviation_pct": dev,
                    "backstep": backstep,
                    "spacing_type": st,
                    "spacing": sv,
                    "max_layers": ml,
                    "tp_multiple": tm,
                    **r,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No completed trades. Broaden the strategy grid.")

    out = out.sort_values(
        ["avg_monthly_return_pct", "total_return_pct"],
        ascending=False,
    )
    out.to_csv(OUT / "top_setups.csv", index=False)
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
