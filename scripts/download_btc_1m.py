#!/usr/bin/env python3
"""
Download the trailing 1 year of BTCUSDT 1-minute spot OHLCV data from Binance Vision.

Output:
  data/BTCUSDT_1m_1y.csv

Columns:
  timestamp,open,high,low,close,volume
Timestamp is UTC milliseconds.
The final dataset is clipped to the exact trailing 365-day window.
"""
from __future__ import annotations

import calendar
import csv
import io
import os
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
OUT = Path("data/BTCUSDT_1m_1y.csv")
START = datetime.now(timezone.utc) - timedelta(days=365)
END = datetime.now(timezone.utc)

HEADERS = {"User-Agent": "Mozilla/5.0 BTC-ZigZag-Level-Scalper data downloader"}

def fetch(url: str) -> bytes:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=60) as r:
        return r.read()

def month_iter(start: datetime, end: datetime):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1

def daily_rows(day: datetime):
    ds = day.strftime("%Y-%m-%d")
    url = (
        f"https://data.binance.vision/data/spot/daily/klines/"
        f"{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{ds}.zip"
    )
    try:
        data = fetch(url)
    except Exception:
        return []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            return list(csv.reader(io.TextIOWrapper(f, encoding="utf-8")))

def monthly_rows(year: int, month: int):
    ym = f"{year:04d}-{month:02d}"
    url = (
        f"https://data.binance.vision/data/spot/monthly/klines/"
        f"{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{ym}.zip"
    )
    try:
        data = fetch(url)
    except Exception:
        return None
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            return list(csv.reader(io.TextIOWrapper(f, encoding="utf-8")))

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    start_ms, end_ms = to_ms(START), to_ms(END)

    rows = []
    seen = set()

    for year, month in month_iter(START, END):
        chunk = monthly_rows(year, month)
        if chunk is not None:
            rows.extend(chunk)
            print(f"monthly {year:04d}-{month:02d}: {len(chunk):,} rows")
        else:
            first = datetime(year, month, 1, tzinfo=timezone.utc)
            days = calendar.monthrange(year, month)[1]
            last_day = min(first + timedelta(days=days - 1), END)
            day = first
            while day <= last_day:
                if day >= START:
                    chunk = daily_rows(day)
                    rows.extend(chunk)
                    print(f"daily {day:%Y-%m-%d}: {len(chunk):,} rows")
                day += timedelta(days=1)

    cleaned = {}
    for r in rows:
        if len(r) < 6:
            continue
        try:
            ts = int(float(r[0]))
        except ValueError:
            continue
        if start_ms <= ts <= end_ms:
            # Binance kline CSV: open time, open, high, low, close, volume, ...
            cleaned[ts] = r[:6]

    ordered = [cleaned[k] for k in sorted(cleaned)]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        w.writerows(ordered)

    print(f"wrote {OUT}: {len(ordered):,} candles")
    if not ordered:
        raise RuntimeError("No candles downloaded.")

if __name__ == "__main__":
    main()
