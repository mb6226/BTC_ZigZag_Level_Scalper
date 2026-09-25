#!/usr/bin/env python3
"""
Download the trailing 1 year of BTCUSDT 1-minute SPOT OHLCV data from Binance Vision.

Output:
  data/BTCUSDT_1m_1y.csv

Binance SPOT kline archives use microsecond timestamps from 2025-01-01 onward.
This script normalizes timestamps to UTC milliseconds in the output.
"""
from __future__ import annotations

import calendar
import csv
import io
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
OUT = Path("data/BTCUSDT_1m_1y.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 BTC-ZigZag-Level-Scalper"}

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

def read_zip_csv(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            return list(csv.reader(io.TextIOWrapper(f, encoding="utf-8")))

def daily_rows(day: datetime):
    ds = day.strftime("%Y-%m-%d")
    url = f"https://data.binance.vision/data/spot/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{ds}.zip"
    try:
        return read_zip_csv(fetch(url))
    except (HTTPError, URLError, zipfile.BadZipFile):
        return []

def monthly_rows(year: int, month: int):
    ym = f"{year:04d}-{month:02d}"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{ym}.zip"
    try:
        return read_zip_csv(fetch(url))
    except (HTTPError, URLError, zipfile.BadZipFile):
        return None

def normalize_timestamp(raw: str) -> int:
    ts = int(float(raw))
    if ts >= 10**17:      # nanoseconds
        return ts // 1_000_000
    if ts >= 10**14:      # microseconds
        return ts // 1_000
    if ts >= 10**11:      # milliseconds
        return ts
    return ts * 1_000     # seconds

def add_rows(cleaned, rows, start_ms, end_ms):
    for r in rows:
        if len(r) < 6:
            continue
        try:
            ts = normalize_timestamp(r[0])
        except (ValueError, TypeError):
            continue
        if start_ms <= ts <= end_ms:
            cleaned[ts] = [str(ts), *r[1:6]]

def main():
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {}

    for year, month in month_iter(start, now):
        chunk = monthly_rows(year, month)

        if chunk is not None:
            print(f"monthly {year:04d}-{month:02d}: {len(chunk):,} rows")
            add_rows(cleaned, chunk, start_ms, end_ms)
            continue

        first = datetime(year, month, 1, tzinfo=timezone.utc)
        days = calendar.monthrange(year, month)[1]
        last = min(first + timedelta(days=days - 1), now)
        day = first

        while day <= last:
            # Binance daily archives are normally available only through yesterday.
            if start.date() <= day.date() < now.date():
                chunk = daily_rows(day)
                print(f"daily {day:%Y-%m-%d}: {len(chunk):,} rows")
                add_rows(cleaned, chunk, start_ms, end_ms)
            day += timedelta(days=1)

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
