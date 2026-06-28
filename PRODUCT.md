# Product

## Register

product

## Users

Solo quantitative trader (the developer). Uses the screener daily, typically after market close or pre-market. Stares at the UI for 10–30 min per session making position decisions. No public audience — design for the power user, not onboarding.

## Product Purpose

Ranks US equities using institutional-grade momentum and fundamental factors, then overlays short-term technical oscillators and AI-powered news sentiment to produce actionable entry/exit signals. Five distinct screens:

1. **Screener Results** — ranked list of top 20 momentum stocks with composite scores, conviction, streaks, and news entry signals
2. **Market Regime** — SPY-based composite bull/bear/neutral signal from 12+ macro/technical indicators
3. **Open Positions** — live P&L tracking with exit signal monitoring (RSI, MACD, Stoch, ADX, MFI)
4. **Filing Edge** — 10-K/10-Q language-stability screen targeting small/micro-cap neglected stocks (based on Cohen, Malloy & Nguyen 2020 "Lazy Prices")
5. **Confluence** — cross-screen: Filing Edge longs checked against price momentum, producing Strong Buy / Watchlist / Filing Only tiers

## Brand Personality

Precise, analytical, signal-first. Three words: **sharp, minimal, purposeful**. No decoration for its own sake. Every visual element either carries data or reduces cognitive load. Feels like a tool a quant at a hedge fund built for themselves.

## Anti-references

- Generic SaaS dashboards (blue gradients, card grids, metric tiles, sidebar icon rails)
- Streamlit default theme — obvious framework look, no visual identity
- Consumer fintech (Robinhood, Coinbase) — pastel, simplified, hides data
- Crypto dashboard aesthetics — neon, glow-everything, busy

## Design Principles

1. **Data is the decoration** — numbers, tickers, and signals ARE the visual hierarchy. Don't add chrome that competes with data.
2. **Color carries meaning** — green = bull/confirm/positive, amber = caution/wait, red = bear/avoid/exit. No decorative color use.
3. **Dense but breathable** — pack information without overwhelming. Whitespace is earned, not defaulted.
4. **Zero clicks to the signal** — the most important output (top pick + conviction + entry signal) should be visible without scrolling or clicking on any page.
5. **Trust the user** — no hand-holding UX copy. The user knows what momentum, SUE, and ADX mean. Labels should be precise, not explanatory.

## Accessibility & Inclusion

WCAG AA minimum. Dark theme primary (reduces eye strain for long daily sessions). All signal colors must pass 4.5:1 contrast on dark backgrounds. Reduced motion respected for all animations.
