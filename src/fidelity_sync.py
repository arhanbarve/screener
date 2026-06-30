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


def _parse_fidelity_csv(content: str) -> list[dict]:
    """
    Parse Fidelity positions CSV — returns rich dicts with all available fields.
    Skips cash, money-market, and crypto rows.
    """
    lines = content.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Symbol" in line and "Quantity" in line:
            header_idx = i
            break
    if header_idx is None:
        print("Could not find header row in CSV", flush=True)
        return []

    holdings = []
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        symbol = (row.get("Symbol") or "").strip().upper()
        if not symbol or symbol.startswith("--"):
            continue
        # Skip cash / money market / crypto
        if any(x in symbol for x in ["FCASH", "FDRXX", "SPAXX", "FZFXX", "FDIC", "PENDING",
                                       "USD***", "BTC/", "ETH/", "USDC"]):
            continue
        if not symbol.replace(".", "").isalpha():
            continue

        qty = _n(row.get("Quantity") or "0")
        if qty <= 0:
            continue

        holdings.append({
            "ticker":          symbol,
            "description":     (row.get("Description") or "").strip(),
            "account":         (row.get("Account Number") or "").strip(),
            "account_name":    (row.get("Account Name") or "").strip(),
            "quantity":        qty,
            "last_price":      _n(row.get("Last Price")),
            "last_price_chg":  _n(row.get("Last Price Change")),
            "current_value":   _n(row.get("Current Value")),
            "today_gl_dollar": _n(row.get("Today's Gain/Loss Dollar")),
            "today_gl_pct":    _n(row.get("Today's Gain/Loss Percent")) / 100,
            "total_gl_dollar": _n(row.get("Total Gain/Loss Dollar")),
            "total_gl_pct":    _n(row.get("Total Gain/Loss Percent")) / 100,
            "pct_of_account":  _n(row.get("Percent Of Account")) / 100,
            "cost_basis_total":_n(row.get("Cost Basis Total")),
            "avg_cost":        _n(row.get("Average Cost Basis")),
        })

    return holdings


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
            sys.exit(1)

        # ── Step 5: parse CSV ─────────────────────────────────────────────────
        holdings = _parse_fidelity_csv(csv_content)
        if not holdings:
            _notify("Fidelity Sync", "No equity holdings found in CSV")
            print("WARNING: parsed 0 holdings", flush=True)
            sys.exit(0)

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


def main() -> None:
    asyncio.run(run_sync())


if __name__ == "__main__":
    main()
