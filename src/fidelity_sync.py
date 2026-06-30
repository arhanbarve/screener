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
from datetime import date
from pathlib import Path

SCREENER_DIR = Path(__file__).parent.parent
POSITIONS_FILE = SCREENER_DIR / "positions.json"
DOWNLOAD_DIR   = SCREENER_DIR / "data" / "fidelity"
LOGIN_TIMEOUT  = 180_000  # ms — 3 minutes to log in + 2FA

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


def _parse_fidelity_csv(content: str) -> list[dict]:
    """
    Parse Fidelity positions CSV export.

    Fidelity prepends account-info rows before the actual table headers.
    We scan for the row that contains 'Symbol' to find the real header.
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
        if not symbol or symbol.startswith("--") or not symbol.replace(".", "").isalpha():
            continue
        # Skip cash / money market positions
        if any(x in symbol for x in ["FCASH", "FDRXX", "SPAXX", "FZFXX", "FDIC", "PENDING"]):
            continue

        try:
            qty = float((row.get("Quantity") or "0").replace(",", "").replace("$", "") or "0")
        except ValueError:
            qty = 0.0
        if qty <= 0:
            continue

        raw_cost = (
            row.get("Average Cost Basis")
            or row.get("Cost Basis Per Share")
            or "0"
        ).replace(",", "").replace("$", "").strip()
        try:
            avg_cost = float(raw_cost or "0")
        except ValueError:
            avg_cost = 0.0

        holdings.append({"ticker": symbol, "avg_cost": avg_cost, "quantity": qty})

    return holdings


# ── main async workflow ───────────────────────────────────────────────────────

async def run_sync() -> None:
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    _notify("Fidelity Sync", "Log in to Fidelity — positions will sync automatically")

    async with async_playwright() as pw:
        # Prefer system Chrome (keeps user's profile); fall back to Chromium
        try:
            browser = await pw.chromium.launch(
                headless=False,
                channel="chrome",
                args=["--start-maximized"],
            )
        except Exception:
            browser = await pw.chromium.launch(
                headless=False,
                args=["--start-maximized"],
            )

        ctx = await browser.new_context(accept_downloads=True, viewport=None)
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

        # ── Step 3: navigate to Positions tab without a hardcoded goto URL ────
        # Click "Positions" nav link if not already there; avoid goto() which
        # sends a bare HTTP request that gets S3 Access Denied without cookies
        if "positions" not in page.url.lower():
            for sel in [
                "a:has-text('Positions')",
                "li:has-text('Positions') a",
                "[data-testid*='positions' i]",
                "button:has-text('Positions')",
            ]:
                try:
                    el = page.locator(sel).first
                    if await el.is_visible(timeout=3_000):
                        await el.click()
                        await page.wait_for_timeout(4_000)
                        print(f"Clicked Positions via: {sel}", flush=True)
                        break
                except Exception:
                    continue

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

        # ── Step 5: click Download button ────────────────────────────────────
        csv_content = None
        download_path = DOWNLOAD_DIR / f"fidelity_positions_{today}.csv"

        for sel in [
            "button:has-text('Download')",
            "a:has-text('Download')",
            "button[aria-label*='Download' i]",
            "a[aria-label*='Download' i]",
            "[data-testid*='download' i]",
            "button:has-text('Export')",
            "a:has-text('Export')",
            "button:has-text('export')",
        ]:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=2_000):
                    async with page.expect_download(timeout=15_000) as dl_info:
                        await btn.click()
                    dl = await dl_info.value
                    await dl.save_as(str(download_path))
                    csv_content = download_path.read_text()
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
