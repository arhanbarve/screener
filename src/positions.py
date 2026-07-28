import json
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

POSITIONS_FILE = Path("positions.json")


def load_positions() -> list[dict]:
    """Load positions from positions.json. Returns [] if file missing."""
    if not POSITIONS_FILE.exists():
        return []
    with open(POSITIONS_FILE, "r") as f:
        return json.load(f)


def save_positions(positions: list[dict]) -> None:
    """Atomically write positions to positions.json."""
    tmp = POSITIONS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(positions, f, indent=2)
    os.replace(tmp, POSITIONS_FILE)


def fetch_price_on_date(ticker: str, entry_date: str) -> float | None:
    """Fetch closing price for ticker on or before entry_date. Returns None on failure."""
    try:
        dt = datetime.strptime(entry_date, "%Y-%m-%d")
        start = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=2)).strftime("%Y-%m-%d")
        raw = yf.download(ticker, start=start, end=end, interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.dropna(how="all")
        if df.empty:
            return None
        target = pd.Timestamp(entry_date)
        available = df.index[df.index <= target]
        if available.empty:
            return float(df["close"].iloc[0])
        return float(df.loc[available[-1], "close"])
    except Exception:
        return None


def add_position(ticker: str, entry_date: str, entry_price: float | None = None) -> float:
    """Append a new position. Auto-fetches close price on entry_date if not provided.
    Returns the entry price used. Raises ValueError if ticker already open or price unavailable."""
    positions = load_positions()
    if any(p["ticker"] == ticker.upper() for p in positions):
        raise ValueError(f"{ticker.upper()} already in open positions")
    if entry_price is None:
        entry_price = fetch_price_on_date(ticker, entry_date)
        if entry_price is None:
            raise ValueError(f"Could not fetch price for {ticker.upper()} on {entry_date}")
    positions.append({
        "ticker": ticker.upper(),
        "entry_date": entry_date,
        "entry_price": float(entry_price),
    })
    save_positions(positions)
    return float(entry_price)


def remove_position(ticker: str) -> None:
    """Remove a position by ticker. No-op if not found."""
    positions = [p for p in load_positions() if p["ticker"] != ticker.upper()]
    save_positions(positions)


def fetch_ohlcv(ticker: str, days: int = 60) -> pd.DataFrame:
    """Fetch OHLCV via yfinance. Returns empty DataFrame on failure."""
    try:
        raw = yf.download(
            ticker,
            period=f"{days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        # Flatten MultiIndex columns if present (single-ticker download)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def get_live_quote(ticker: str) -> tuple[float | None, float | None]:
    """Fetch latest price and previous close via yfinance fast_info. (None, None) on failure."""
    try:
        info = yf.Ticker(ticker).fast_info
        last = float(info.last_price)
        try:
            prev = float(info.previous_close)
        except Exception:
            prev = None
        return last, prev
    except Exception:
        return None, None




def days_to_next_earnings(ticker: str) -> int | None:
    """Calendar days until the next scheduled earnings date. None on failure.

    Reuses the yfinance earnings-dates endpoint (same source as
    src/pead_backtest.py). Best-effort: returns None if the call fails or no
    future date is listed.
    """
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if df is None or df.empty:
            return None
        today = pd.Timestamp.now(tz=df.index.tz) if df.index.tz else pd.Timestamp.now()
        future = [ts for ts in df.index if ts >= today]
        if not future:
            return None
        return int((min(future) - today).days)
    except Exception:
        return None
