"""
Fidelity positions sync.

Launched on morning login by sync_fidelity.sh (launchd RunAtLoad).
Opens Chrome → navigates to Fidelity login → waits for user to log in
→ downloads positions CSV → merges into positions.json.
"""
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

SCREENER_DIR       = Path(__file__).parent.parent
POSITIONS_FILE     = SCREENER_DIR / "positions.json"
FIDELITY_DATA_FILE = SCREENER_DIR / "data" / "fidelity" / "positions_data.json"
DOWNLOAD_DIR       = SCREENER_DIR / "data" / "fidelity"
LOGIN_TIMEOUT      = 180_000  # ms — 3 minutes to log in + 2FA

FIDELITY_LOGIN = "https://login.fidelity.com/ftgw/Fidelity/RtlCust/Login/Init"

STATUS_FILE = SCREENER_DIR / "logs" / "fidelity_sync_status.json"


def _write_status(result: str, message: str, positions_synced: int | None = None) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": date.today().isoformat(),
        "attempted_at": datetime.now().isoformat(timespec="minutes"),
        "result": result,
        "message": message,
        "positions_synced": positions_synced,
    }
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, STATUS_FILE)


# ── helpers ───────────────────────────────────────────────────────────────────

def _notify(title: str, message: str) -> None:
    subprocess.run(
        ["osascript", "-e",
         f'display notification "{message}" with title "{title}"'],
        check=False,
    )


def _load_positions() -> list[dict]:
    if POSITIONS_FILE.exists():
        return json.loads(POSITIONS_FILE.read_text())
    return []


def _save_positions(positions: list[dict]) -> None:
    tmp = POSITIONS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(positions, indent=2))
    os.replace(tmp, POSITIONS_FILE)


def _n(s: str) -> float:
    """Strip currency/sign/percent chars and return float. Handles '62062.06 / BTC'."""
    s = (s or "").replace("$", "").replace(",", "").replace("+", "").replace("%", "").strip()
    s = s.split("/")[0].strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


class FidelityCsvSchemaError(RuntimeError):
    """The Fidelity CSV no longer exposes the columns we depend on.

    Raised instead of returning zeros. In Jul 2026 Fidelity renamed every
    multi-word header from Title Case to Sentence case ("Last Price" ->
    "Last price"). csv.DictReader keys are exact, so every money lookup
    returned None, _n(None) returned 0.0, and the sync reported success while
    writing an all-zero portfolio — the Positions page then showed Total G/L
    equal to the entire account value for five days. Silence was the whole bug;
    this exception exists so the next rename is loud.
    """


def _norm_key(s: str) -> str:
    """Reduce a CSV header to alphanumerics so case/spacing/punctuation churn
    cannot break the column mapping. Also drops any UTF-8 BOM, which the
    response-interception path can leave on the first field name.

    "Last Price" / "Last price" / " LAST PRICE " all -> "lastprice"
    """
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# Column -> canonical Fidelity label, used for the mapping and for naming what
# went missing in the error message. Keys are normalised header forms.
_REQUIRED_COLS: dict[str, str] = {
    "symbol":                "Symbol",
    "quantity":              "Quantity",
    "lastprice":             "Last Price",
    "currentvalue":          "Current Value",
    "todaysgainlossdollar":  "Today's Gain/Loss Dollar",
    "todaysgainlosspercent": "Today's Gain/Loss Percent",
    "totalgainlossdollar":   "Total Gain/Loss Dollar",
    "totalgainlosspercent":  "Total Gain/Loss Percent",
    "percentofaccount":      "Percent Of Account",
    "costbasistotal":        "Cost Basis Total",
    "averagecostbasis":      "Average Cost Basis",
}

# Nice to have, not worth aborting a sync over — it is display-only.
_OPTIONAL_COLS: dict[str, str] = {
    "lastpricechange": "Last Price Change",
}


def _parse_fidelity_csv(content: str) -> list[dict]:
    """
    Parse Fidelity positions CSV — returns rich dicts with all available fields.
    Skips cash, money-market, and crypto rows.

    Raises FidelityCsvSchemaError if the header is unrecognisable or a required
    money column is absent. Never returns rows with silently zeroed fields.
    """
    lines = content.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        norm = _norm_key(line)
        if "symbol" in norm and "quantity" in norm:
            header_idx = i
            break
    if header_idx is None:
        raise FidelityCsvSchemaError(
            "Could not find a header row containing Symbol and Quantity — "
            "the download is not a positions export."
        )

    reader = csv.DictReader(lines[header_idx:])
    fieldnames = reader.fieldnames or []

    # Map normalised header -> the exact key DictReader will use for that column.
    by_norm = {_norm_key(f): f for f in fieldnames if f is not None}

    missing = [
        label for norm, label in _REQUIRED_COLS.items() if norm not in by_norm
    ]
    if missing:
        raise FidelityCsvSchemaError(
            "Fidelity CSV is missing required column(s): "
            + ", ".join(missing)
            + f". Header was: {', '.join(str(f) for f in fieldnames)}"
        )

    # Optional columns don't abort the sync, but a disappearing one is an early
    # warning that Fidelity is churning the export again — say so in the log.
    for norm, label in _OPTIONAL_COLS.items():
        if norm not in by_norm:
            print(f"NOTE: optional column '{label}' absent from Fidelity CSV", flush=True)

    def _col(row: dict, norm: str) -> str:
        """Read a column by its normalised name. Optional columns may be absent."""
        key = by_norm.get(norm)
        return "" if key is None else (row.get(key) or "")

    holdings = []
    for row in reader:
        symbol = _col(row, "symbol").strip().upper()
        if not symbol or symbol.startswith("--"):
            continue
        # Skip cash / money market / crypto
        if any(x in symbol for x in ["FCASH", "FDRXX", "SPAXX", "FZFXX", "FDIC", "PENDING",
                                       "USD***", "BTC/", "ETH/", "USDC"]):
            continue
        if not symbol.replace(".", "").isalpha():
            continue

        qty = _n(_col(row, "quantity") or "0")
        if qty <= 0:
            continue

        # SECURITY: Account Number / Account Name / Description are deliberately
        # NOT captured. This file is committed to git so the cloud dashboard can
        # read it, and nothing in the app consumes those three fields — they were
        # pure identifying data sitting in a tracked file. Do not add them back;
        # scripts/pre-commit blocks account-number-shaped strings anyway.
        holdings.append({
            "ticker":          symbol,
            "quantity":        qty,
            "last_price":      _n(_col(row, "lastprice")),
            "last_price_chg":  _n(_col(row, "lastpricechange")),
            "current_value":   _n(_col(row, "currentvalue")),
            "today_gl_dollar": _n(_col(row, "todaysgainlossdollar")),
            "today_gl_pct":    _n(_col(row, "todaysgainlosspercent")) / 100,
            "total_gl_dollar": _n(_col(row, "totalgainlossdollar")),
            "total_gl_pct":    _n(_col(row, "totalgainlosspercent")) / 100,
            "pct_of_account":  _n(_col(row, "percentofaccount")) / 100,
            "cost_basis_total":_n(_col(row, "costbasistotal")),
            "avg_cost":        _n(_col(row, "averagecostbasis")),
        })

    return holdings


def _looks_corrupt(holdings: list[dict]) -> bool:
    """True when a parse succeeded structurally but produced unusable values.

    The header can stay intact while the *values* stop parsing — a currency or
    locale change, or a column that starts coming through blank. Real equity
    holdings always have a price and a cost basis somewhere, so an all-zero
    parse is corruption, not a portfolio, and must not overwrite the last good
    snapshot.
    """
    if not holdings:
        return False  # emptiness is handled separately — that's "no holdings"
    has_price = any(h["last_price"] > 0 for h in holdings)
    has_basis = any(
        h["avg_cost"] > 0 or h["cost_basis_total"] > 0 for h in holdings
    )
    return not (has_price and has_basis)


def _save_fidelity_data(holdings: list[dict]) -> None:
    FIDELITY_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "synced_at": datetime.now().isoformat(timespec="minutes"),
        "positions": holdings,
    }
    tmp = FIDELITY_DATA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, FIDELITY_DATA_FILE)


# ── main async workflow ───────────────────────────────────────────────────────

async def run_sync() -> None:
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    _notify("Fidelity Sync", "Log in to Fidelity — positions will sync automatically")
    _write_status("attempted", "Waiting for login")

    async with async_playwright() as pw:
        stealth_args = [
            "--disable-blink-features=AutomationControlled",
            "--start-maximized",
            "--no-first-run",
            "--no-service-autorun",
            "--password-store=basic",
        ]
        # Removing "--enable-automation" prevents Fidelity's bot detection from
        # seeing the automation flag that Playwright injects by default.
        try:
            browser = await pw.chromium.launch(
                headless=False,
                channel="chrome",
                args=stealth_args,
                ignore_default_args=["--enable-automation"],
            )
        except Exception:
            browser = await pw.chromium.launch(
                headless=False,
                args=stealth_args,
                ignore_default_args=["--enable-automation"],
            )

        ctx = await browser.new_context(
            accept_downloads=True,
            viewport=None,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/137.0.0.0 Safari/537.36"
            ),
        )
        # Patch navigator.webdriver so JS-based bot detection sees undefined
        await ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        page = await ctx.new_page()

        # Intercept CSV responses anywhere on the page so we don't have to
        # find the exact download button — just let the browser handle it
        intercepted_csv: list[str] = []

        async def _intercept(response) -> None:
            ct = response.headers.get("content-type", "")
            if "csv" in ct or "text/plain" in ct or "octet-stream" in ct:
                try:
                    text = await response.text()
                    if "Symbol" in text and "Quantity" in text:
                        intercepted_csv.append(text)
                        print(f"Intercepted CSV from {response.url}", flush=True)
                except Exception:
                    pass

        page.on("response", _intercept)

        # ── Step 1: open login page ───────────────────────────────────────────
        await page.goto(FIDELITY_LOGIN, wait_until="domcontentloaded")
        print("Waiting for you to log in…", flush=True)

        # ── Step 2: wait for fully authenticated portfolio page ───────────────
        # Only match FINAL post-login destination; exclude auth/redirect URLs
        def _is_authenticated(url: str) -> bool:
            if any(x in url for x in ["login", "user-identity", "authentication", "mfa", "2fa", "challenge"]):
                return False
            return "digital.fidelity.com" in url and any(
                k in url for k in ["/portfolio", "/summary", "/oltx", "/mymoney", "/accounts"]
            )

        try:
            await page.wait_for_url(_is_authenticated, timeout=LOGIN_TIMEOUT)
            # Extra settle time — let the SPA fully boot before we do anything
            await page.wait_for_timeout(4_000)
        except PWTimeout:
            _notify("Fidelity Sync", "Login timed out — using last known positions")
            _write_status("timeout", "Login timed out — using last known positions")
            await browser.close()
            sys.exit(0)

        _notify("Fidelity Sync", "Logged in — looking for Positions…")
        print(f"Authenticated. Current URL: {page.url}", flush=True)

        # ── Step 3: navigate to Positions page ───────────────────────────────
        # We are authenticated — session cookies exist in this browser context,
        # so page.goto() will carry them and NOT trigger S3 Access Denied.
        POSITIONS_URL = "https://digital.fidelity.com/ftgw/digital/portfolio/positions"
        if "positions" not in page.url.lower():
            # Try clicking the nav link first (cleanest SPA navigation)
            clicked = False
            for sel in [
                "a[href*='/portfolio/positions']",
                "a:has-text('Positions')",
                "a:has-text('Account Positions')",
            ]:
                try:
                    el = page.locator(sel).first
                    if await el.is_visible(timeout=3_000):
                        await el.click()
                        await page.wait_for_timeout(6_000)
                        print(f"Clicked Positions via: {sel}, now at: {page.url}", flush=True)
                        clicked = True
                        break
                except Exception:
                    continue

            # Fallback: direct goto (works fine once authenticated)
            if not clicked or "positions" not in page.url.lower():
                print(f"Nav click failed or URL wrong ({page.url}), using goto", flush=True)
                await page.goto(POSITIONS_URL, wait_until="load", timeout=30_000)
                await page.wait_for_timeout(8_000)  # SPA render time
                print(f"After goto: {page.url}", flush=True)

        # ── Step 4: take diagnostic screenshot AFTER reaching positions ────────
        diag_dir = DOWNLOAD_DIR / "diag"
        diag_dir.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(diag_dir / "positions_page.png"), full_page=True)
        btns = await page.evaluate("""() => {
            const els = document.querySelectorAll('button, a, [role=button]');
            return Array.from(els).map(e => ({
                tag: e.tagName,
                text: (e.innerText||'').trim().slice(0,80),
                ariaLabel: e.getAttribute('aria-label')||'',
                id: e.id||'',
                href: e.href||'',
            })).filter(e => e.text||e.ariaLabel);
        }""")
        import json as _json
        (diag_dir / "buttons.json").write_text(_json.dumps(btns, indent=2))
        print(f"Diagnostic: {len(btns)} elements, screenshot saved", flush=True)

        # ── Step 5: open kebab menu then click Download ───────────────────────
        csv_content = None
        download_path = DOWNLOAD_DIR / f"fidelity_positions_{today}.csv"

        # The Download button lives inside the "Available Actions" kebab menu.
        # Open it first, then click the menu item.
        try:
            kebab = page.locator("#button-984246106365, button:has-text('Available Actions')").first
            if await kebab.is_visible(timeout=3_000):
                await kebab.click()
                await page.wait_for_timeout(1_000)
                print("Opened Available Actions menu", flush=True)
        except Exception as e:
            print(f"Could not open kebab menu: {e}", flush=True)

        for sel in [
            "#kebabmenuitem-download",
            "button:has-text('Download')",
            "a:has-text('Download')",
            "button[aria-label*='Download' i]",
            "[data-testid*='download' i]",
            "button:has-text('Export')",
        ]:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=2_000):
                    async with page.expect_download(timeout=15_000) as dl_info:
                        await btn.click()
                    dl = await dl_info.value
                    await dl.save_as(str(download_path))
                    csv_content = download_path.read_text(encoding='utf-8-sig')
                    print(f"Downloaded via selector: {sel}", flush=True)
                    break
            except Exception:
                continue

        # Fallback: use whatever the response interceptor caught
        if not csv_content and intercepted_csv:
            csv_content = intercepted_csv[-1]
            download_path.write_text(csv_content)
            print("Used intercepted CSV response", flush=True)

        await browser.close()

        if not csv_content:
            _notify("Fidelity Sync", "Could not download CSV — see data/fidelity/diag/")
            print("ERROR: no CSV. Check data/fidelity/diag/positions_page.png and buttons.json", flush=True)
            _write_status("no_csv", "Could not download CSV")
            sys.exit(1)

        # ── Step 5: parse CSV ─────────────────────────────────────────────────
        # A schema change must abort loudly and leave the previous good snapshot
        # in place. Overwriting positions_data.json with zeros is strictly worse
        # than keeping yesterday's real numbers.
        try:
            holdings = _parse_fidelity_csv(csv_content)
        except FidelityCsvSchemaError as e:
            _notify("Fidelity Sync", "CSV format changed — keeping last good data")
            print(f"ERROR: {e}", flush=True)
            _write_status("bad_header", str(e))
            sys.exit(1)

        if not holdings:
            _notify("Fidelity Sync", "No equity holdings found in CSV")
            print("WARNING: parsed 0 holdings", flush=True)
            _write_status("no_holdings", "No equity holdings found in CSV")
            sys.exit(0)

        # Second guard: structurally valid header, unusable values.
        if _looks_corrupt(holdings):
            msg = (
                f"Parsed {len(holdings)} holdings but every price/cost basis was "
                f"zero — refusing to overwrite last good snapshot"
            )
            _notify("Fidelity Sync", "CSV values unparseable — keeping last good data")
            print(f"ERROR: {msg}", flush=True)
            _write_status("bad_values", msg)
            sys.exit(1)

        print(f"Parsed {len(holdings)} holdings from Fidelity", flush=True)

        # Save rich data for the UI to read
        _save_fidelity_data(holdings)

        # Deduplicate by ticker (keep first; user may hold same stock in multiple accounts)
        seen: set[str] = set()
        deduped: list[dict] = []
        for h in holdings:
            if h["ticker"] not in seen:
                seen.add(h["ticker"])
                deduped.append(h)
        holdings = deduped

        # ── Step 6: merge into positions.json ────────────────────────────────
        fidelity_tickers  = {h["ticker"] for h in holdings}
        current_positions = _load_positions()
        current_tickers   = {p["ticker"] for p in current_positions}

        # Keep positions still held; drop positions sold in Fidelity
        updated = [p for p in current_positions if p["ticker"] in fidelity_tickers]
        removed = current_tickers - fidelity_tickers

        # Add newly detected Fidelity holdings not yet in screener
        added = []
        for h in holdings:
            if h["ticker"] not in current_tickers:
                updated.append({
                    "ticker":      h["ticker"],
                    "entry_date":  today,
                    "entry_price": round(h["avg_cost"], 4) if h["avg_cost"] > 0 else 0.0,
                })
                added.append(h["ticker"])

        _save_positions(updated)

        parts = [f"Synced {len(updated)} positions"]
        if added:   parts.append(f"Added: {', '.join(added)}")
        if removed: parts.append(f"Removed: {', '.join(removed)}")
        msg = " · ".join(parts)
        _notify("Fidelity Sync", msg)
        print(msg, flush=True)
        result = "no_change" if not added and not removed else "success"
        _write_status(result, msg, positions_synced=len(updated))


def main() -> None:
    try:
        asyncio.run(run_sync())
    except SystemExit:
        raise
    except Exception as e:
        _write_status("failed", str(e))
        raise


if __name__ == "__main__":
    main()
