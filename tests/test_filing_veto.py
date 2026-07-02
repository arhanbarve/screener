"""Tests for the filing-edge veto overlay."""

import json

import pandas as pd

from src.filing_veto import (
    build_veto_report,
    load_position_tickers,
    load_screener_tickers,
    load_watch,
)


def _write_fixtures(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    watch_df = pd.DataFrame(
        [
            {
                "ticker": "FOO",
                "__list__": "watch",
                "text_stability": 0.42,
                "change_direction": -1,
            },
            {
                "ticker": "IGN",
                "__list__": "watch",
                "text_stability": 0.99,
                "change_direction": 0,
            },
            {
                "ticker": "LNG",
                "__list__": "long",
                "text_stability": 0.98,
                "change_direction": 1,
            },
        ]
    )
    watch_df.to_csv(output_dir / "filing_edge_2026-07-01.csv", index=False)

    screen_df = pd.DataFrame([{"ticker": "FOO"}, {"ticker": "XYZ"}])
    screen_df.to_csv(output_dir / "screen_2026-07-01.csv", index=False)

    positions_path = tmp_path / "positions.json"
    positions_path.write_text(
        json.dumps([{"ticker": "BAR", "entry_date": "2026-06-01", "entry_price": 10.0}])
    )
    return output_dir, positions_path


def test_build_veto_report_flags_screener_conflict(tmp_path):
    output_dir, positions_path = _write_fixtures(tmp_path)

    report = build_veto_report(
        output_dir=str(output_dir), positions_path=str(positions_path)
    )

    # FOO is on the watch list AND in the screener -> actionable conflict.
    assert "FOO" in set(report["ticker"])
    foo = report[report["ticker"] == "FOO"].iloc[0]
    assert bool(foo["in_screener"]) is True
    assert bool(foo["in_positions"]) is False
    assert foo["change_direction"] == -1

    # IGN is watch-listed but not screened/held -> excluded.
    assert "IGN" not in set(report["ticker"])
    # LNG is on the long list -> never considered.
    assert "LNG" not in set(report["ticker"])


def test_build_veto_report_flags_held_position(tmp_path):
    output_dir, positions_path = _write_fixtures(tmp_path)

    # Add BAR (a held position) to the watch list.
    watch_path = output_dir / "filing_edge_2026-07-01.csv"
    df = pd.read_csv(watch_path)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "ticker": "BAR",
                        "__list__": "watch",
                        "text_stability": 0.55,
                        "change_direction": 0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df.to_csv(watch_path, index=False)

    report = build_veto_report(
        output_dir=str(output_dir), positions_path=str(positions_path)
    )
    bar = report[report["ticker"] == "BAR"].iloc[0]
    assert bool(bar["in_positions"]) is True
    assert bool(bar["in_screener"]) is False

    # Deteriorating filer (FOO, change_direction == -1) sorts first.
    assert report.iloc[0]["ticker"] == "FOO"


def test_loaders_robust_to_missing_files(tmp_path):
    empty_dir = tmp_path / "nothing"
    empty_dir.mkdir()

    assert load_watch(output_dir=str(empty_dir)).empty
    assert load_screener_tickers(output_dir=str(empty_dir)) == set()
    assert load_position_tickers(str(tmp_path / "missing.json")) == set()

    report = build_veto_report(
        output_dir=str(empty_dir), positions_path=str(tmp_path / "missing.json")
    )
    assert report.empty
    assert list(report.columns) == [
        "ticker",
        "in_screener",
        "in_positions",
        "text_stability",
        "change_direction",
        "change_reason",
        "eight_k_penalty",
        "red_flags",
    ]
