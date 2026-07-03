"""Shared recipe steps (docs/strategy-specs.md "Shared recipe"), reused by
every idea in the bank from #2 onward — universe join, dedup, earnings-
contamination filter, targeted earnings checkpoint. Idea #1
(src/sc13d_backtest.py) had its own copies of these before this module
existed; not retrofitted, working code left alone.
"""

import sqlite3

import pandas as pd

from src.pead_backtest import collect_earnings_events


def filter_to_universe(events: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """CIK-join to the >=$2.5B universe (shared recipe step 2). `uni` must
    have zero-padded 10-digit `cik` and `ticker` columns."""
    if events.empty:
        return events
    merged = events.merge(uni[["cik", "ticker"]], on="cik", how="inner")
    return merged.reset_index(drop=True)


def dedupe_events(events: pd.DataFrame, dedupe_days: int = 20) -> pd.DataFrame:
    """One event per ticker per `dedupe_days`; first (earliest) wins
    (shared recipe step 5). Requires `ticker` and `file_date` columns."""
    if events.empty:
        return events
    kept = []
    for ticker, g in events.sort_values("file_date").groupby("ticker"):
        last_kept = None
        for _, row in g.iterrows():
            if last_kept is not None and (row["file_date"] - last_kept).days < dedupe_days:
                continue
            kept.append(row)
            last_kept = row["file_date"]
    return pd.DataFrame(kept).reset_index(drop=True) if kept else events.iloc[0:0]


def drop_earnings_contamination(events: pd.DataFrame, earnings: pd.DataFrame,
                                window_days: int = 3) -> pd.DataFrame:
    """Drop events within `window_days` calendar days of that ticker's
    nearest earnings announcement (shared recipe step 4) — otherwise the
    category measures earnings reaction, not the event itself. Requires
    `ticker`/`event_date` on events, `ticker`/`announce_date` on earnings."""
    if events.empty or earnings.empty:
        return events
    ann_by_ticker = {}
    for ticker, g in earnings.groupby("ticker"):
        ann_by_ticker[ticker] = pd.to_datetime(g["announce_date"]).tolist()

    keep = []
    for _, row in events.iterrows():
        dates = ann_by_ticker.get(row["ticker"])
        if not dates:
            keep.append(True)
            continue
        ev_ts = pd.Timestamp(row["event_date"])
        contaminated = any(abs((ev_ts - d).days) <= window_days for d in dates)
        keep.append(not contaminated)
    return events[pd.Series(keep, index=events.index)].reset_index(drop=True)


def load_earnings_for_tickers(tickers: set, db_path: str) -> pd.DataFrame:
    """Targeted earnings-date checkpoint: fetch only tickers that actually
    appear as event subjects (cheap) rather than the whole universe."""
    if not tickers:
        return pd.DataFrame(columns=["ticker", "announce_date"])
    collect_earnings_events(sorted(tickers), db_path=db_path)
    con = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in tickers)
    df = pd.read_sql_query(
        f"SELECT ticker, announce_date FROM pead_earnings WHERE ticker IN ({placeholders})",
        con, params=list(tickers))
    con.close()
    return df


def split_by_earnings(events: pd.DataFrame, earnings: pd.DataFrame,
                      window_days: int = 3,
                      date_col: str = "event_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like drop_earnings_contamination, but returns BOTH sides:
    (clean, contaminated). Reversal-family ideas use the contaminated rows
    as a control category instead of discarding them."""
    if events.empty:
        return events, events
    if earnings.empty:
        return events.reset_index(drop=True), events.iloc[0:0]
    ann_by_ticker = {}
    for ticker, g in earnings.groupby("ticker"):
        ann_by_ticker[ticker] = pd.to_datetime(g["announce_date"]).tolist()
    contaminated_mask = []
    for _, row in events.iterrows():
        dates = ann_by_ticker.get(row["ticker"], [])
        ev_ts = pd.Timestamp(row[date_col])
        contaminated_mask.append(
            any(abs((ev_ts - d).days) <= window_days for d in dates))
    mask = pd.Series(contaminated_mask, index=events.index)
    return (events[~mask].reset_index(drop=True),
            events[mask].reset_index(drop=True))


def split_by_news(events: pd.DataFrame, filing_dates_by_ticker: dict,
                  days_before: int = 3, days_after: int = 1,
                  date_col: str = "trigger_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(no_news, news) split: an event is 'news' if the ticker filed
    anything in filing_dates_by_ticker within [date - days_before,
    date + days_after]. Tickers absent from the dict count as no-news
    (their filings were checked and none matched, or they have no CIK —
    callers must pass an entry per ticker they actually resolved)."""
    if events.empty:
        return events, events
    news_mask = []
    for _, row in events.iterrows():
        dates = filing_dates_by_ticker.get(row["ticker"], [])
        ev_ts = pd.Timestamp(row[date_col])
        news_mask.append(any(
            -days_before <= (d - ev_ts).days <= days_after for d in dates))
    mask = pd.Series(news_mask, index=events.index)
    return (events[~mask].reset_index(drop=True),
            events[mask].reset_index(drop=True))


def attach_cap_proxy(events: pd.DataFrame, uni: pd.DataFrame,
                     price_cache: dict,
                     date_col: str = "event_date") -> pd.DataFrame:
    """cap_proxy = current market_cap x price_at_event / current_price.
    Pre-registered survivorship mitigation (registry IDEA BANK 2 preamble):
    reversal-family signals must not admit events that were sub-$2.5B when
    they happened just because the name later grew into the universe.
    Rows with no usable price history get NaN (caller filters + logs)."""
    if events.empty:
        events = events.copy()
        events["cap_proxy"] = pd.Series(dtype=float)
        return events
    caps_now = dict(zip(uni["ticker"], uni["market_cap"]))
    vals = []
    for _, row in events.iterrows():
        p = price_cache.get(row["ticker"])
        cap_now = caps_now.get(row["ticker"])
        if p is None or p.empty or cap_now is None:
            vals.append(float("nan"))
            continue
        ts = pd.Timestamp(row[date_col])
        upto = p.loc[p.index <= ts, "close"]
        p_now = float(p["close"].iloc[-1])
        if upto.empty or p_now == 0:
            vals.append(float("nan"))
            continue
        vals.append(float(cap_now) * float(upto.iloc[-1]) / p_now)
    events = events.copy()
    events["cap_proxy"] = vals
    return events


def filter_cap_proxy(events: pd.DataFrame, min_cap: float = 2.5e9) -> pd.DataFrame:
    """Keep rows whose cap_proxy is present and >= min_cap."""
    if events.empty:
        return events
    return events[events["cap_proxy"].notna()
                  & (events["cap_proxy"] >= min_cap)].reset_index(drop=True)
