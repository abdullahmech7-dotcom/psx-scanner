"""
Smart Scanner: Pakistan Stock Exchange + Crypto + Forex (works on a phone browser).
1) pick market  2) pick Intraday or Swing + how long you will hold  3) analyse one symbol, or rank the best opportunities.
Run locally:  streamlit run app.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import psx_scanner as ps

st.set_page_config(page_title="Smart Scanner: PSX, Crypto, Forex", page_icon="📈", layout="wide")
st.title("📈 Smart Scanner: PSX, Crypto & Forex")
st.caption("Educational tool, not financial advice. Always use a stop loss. Past statistics do not guarantee future results.")

MK = {"Pakistan Stock Exchange (PSX)": "PSX", "Crypto": "Crypto", "Forex": "Forex"}
SIGCOL = {"BUY": "color:#16a34a;font-weight:700", "SELL (short)": "color:#dc2626;font-weight:700",
          "EXIT / AVOID": "color:#dc2626;font-weight:700", "SELL/EXIT": "color:#dc2626;font-weight:700", "NO TRADE": "color:#ca8a04"}
GROUPS = {"asset": "This asset", "sector": "Sector / related", "market": "Whole market", "macro": "Economy and central banks", "world": "World events"}


def md(t):
    return str(t).replace("$", "\\$")


# ------------------------------------------------------------------ choices (top of page)
c1, c2, c3 = st.columns(3)
market = MK[c1.selectbox("1. Market", list(MK))]
style = "Intraday" if c2.selectbox("2. Trading style", ["Intraday (close the same day)", "Swing (hold for days)"]).startswith("Intra") else "Swing"
if style == "Intraday":
    opts = [0.5, 1, 2, 3, 4, 6] if market == "PSX" else [0.5, 1, 2, 3, 4, 6, 8, 12]
    hrs = c3.select_slider("3. How long will you hold? (hours)", options=opts, value=2)
    hold_min, hold_txt = hrs * 60, f"{hrs:g} h"
else:
    days = int(c3.number_input("3. How long will you hold? (days)", min_value=1, max_value=60, value=5, step=1))
    hold_min, hold_txt = days * ps.SESSION_MIN[market], f"{days} day(s)"
SIG = (market, style, hold_min)
cur = ps.CURRENCY[market]

mode = st.sidebar.radio("What do you want?", ["Analyse one symbol", "Find best opportunities (ranked)"])
use_news = st.sidebar.checkbox("Include news and events", value=True)
amount = st.sidebar.number_input(f"Example amount ({cur}), only to show profit/loss numbers", min_value=0, value=100000, step=10000)
demo = st.sidebar.checkbox("Demo data (FAKE prices, test only)", value=False)
SIG = SIG + (demo,)
if demo:
    st.error("⚠ DEMO MODE IS ON: all prices below are FAKE, randomly generated. Untick \"Demo data\" in the sidebar for real prices.")
st.sidebar.caption("The trade plan does not depend on your capital. Whatever you invest, the stop loss and targets are the same in % terms.")
if market == "PSX":
    st.sidebar.caption("PSX: only BUY / EXIT ideas (retail short selling is usually impractical).")


# ------------------------------------------------------------------ helpers
def style_signal(df, col="Direction"):
    sty = df.style
    fn = getattr(sty, "map", None) or sty.applymap
    return fn(lambda v: SIGCOL.get(v, ""), subset=[col])


def show_fig(sym, r):
    fig = ps.draw(sym, r, fig_only=True)
    st.pyplot(fig)
    plt.close(fig)


def headlines(items, n=5):
    for i in items[:n]:
        ic = "🟢" if i["s"] > 0 else "🔴" if i["s"] < 0 else "⚪"
        st.write(f"{ic} {md(i['title'])}  ·  {i['src']}, {i['age']:.0f}d")


def render_news(nb, assets=True):
    if not nb:
        st.info("News is switched off (sidebar).")
        return
    if nb.get("err"):
        st.warning(f"News could not be loaded ({nb['err']}). Check the internet connection and retry.")
        return
    if nb["flags"]:
        st.warning("⚠ **Event risk in the last 2 days** (can move prices sharply, even against a good chart):\n\n" +
                   "\n".join(f"- {md(f['title'])} ({f['src']}, keyword: {f['word']})" for f in nb["flags"]))
    for g, lab in GROUPS.items():
        if g not in nb["groups"] or (g in ("asset", "sector") and not assets):
            continue
        items = nb["groups"][g]
        mood, _ = ps.news_label(nb["scores"].get(g, 0))
        st.markdown(f"**{lab}: {mood}** ({nb['scores'].get(g, 0):+.0f})")
        if items:
            headlines(items)
        else:
            st.caption("No headlines in this period.")
    st.caption("Mood is keyword-based. Read the headlines yourself, especially for intraday trades.")


def render_idea(o, key="x", chart=True):
    """One trade idea: headline, plan, checks, news, optional chart."""
    p, dec = o["plan"], o["dec"]
    name = ps.dir_name(o["market"], o["side"])
    if not o["side"]:
        st.info(f"**No clear trade** for a {hold_txt} hold (combined score {o['final']:+.0f}). Waiting is a valid position.")
        return
    box = st.success if (o["valid"] and o["side"] > 0) else st.error if (o["valid"] and o["side"] < 0) else st.warning
    wc = o["bt"]["pw"] * 100 if o["bt"] else 0
    box(f"**{name} {o['sym']}**  |  grade **{o['grade']}**  |  rank score **{o['rank']:.0f}/100**  |  chart timeframe {o['tf']}  |  "
        f"estimated chance of reaching T1 before SL: **{wc:.0f}%** (n={o['bt']['n']} similar past signals)")
    if not o["valid"]:
        st.caption("Not ranked as a trade because: " + "; ".join(o["why_not"]))
    cols = st.columns(4)
    for col, lab, v in zip(cols, ("Entry", "Stop loss", "Target 1", "Target 2"), (p["entry"], p["sl"], p["t1"], p["t2"])):
        col.metric(lab, ps.fp(v, dec), f"{100 * (v - p['entry']) / p['entry']:+.2f}%" if lab != "Entry" else None, delta_color="off")
    st.caption(f"Reward:risk to T1 = {p['tm']:.1f}x  |  T1 expected in {ps.fmt_dur(o['t1_min'], o['market'])}, T2 in {ps.fmt_dur(o['t2_min'], o['market'])}  |  "
               f"**Time-stop:** if T1 is not hit within {hold_txt}, exit." + (f"  |  {o['note']}" if o["note"] else ""))
    if amount:
        rp = 100 * p["risk"] / p["entry"]
        g1, g2 = 100 * abs(p["t1"] - p["entry"]) / p["entry"], 100 * abs(p["t2"] - p["entry"]) / p["entry"]
        st.caption(f"Example with {amount:,.0f} {cur} (any amount works the same): stop-loss loses about {amount * rp / 100:,.0f}, "
                   f"T1 gains about {amount * g1 / 100:,.0f}, T2 about {amount * g2 / 100:,.0f}.")
    for w in o["warn"]:
        st.warning(w)
    if o["checks"]:
        st.write("  ".join(("✅ " if ok else "❌ ") + lab for lab, ok in o["checks"]))
    if o["setups"]:
        st.write("**Intraday setups:** " + "; ".join(f"{n} ({nt})" for n, p_, nt in o["setups"] if p_ * o["side"] > 0))
    pri = o["pri"]
    st.write("**Candles:** " + (", ".join(pri["candles"]) or "none") + "   |   **Chart patterns:** " +
             ("; ".join(f"{n} ({b}) {note}" for n, b, note in pri["chart"]) or "none"))
    if o["news"] and not o["news"].get("err"):
        (st.warning if "CONFLICT" in o["news_note"] else st.caption)(f"News: {o['news_label']} ({o['news_comb']:+.0f}). {o['news_note']}")
        for f in o["flags"][:2]:
            st.caption("Event: " + md(f["title"]))
    if chart:
        show_fig(o["sym"], o["chart_r"])


# ------------------------------------------------------------------ single symbol
def render_single(out):
    res, o, sym = out["results"], out["opp"], out["sym"]
    price = o["px"]
    d1 = next((r for r in res if r["tf"] == "1D"), None)
    chg = ""
    if d1 is not None and len(d1["d"]) > 1:
        c = d1["d"].close
        chg = f"{100 * (c.iloc[-1] / c.iloc[-2] - 1):+.2f}% last daily candle   |   "
    st.subheader(f"{sym}   {ps.fp(price, o['dec'])} {cur if out['market'] == 'PSX' else ''}")
    st.caption(f"{chg}data to {res[0]['d'].index[-1]:%d %b %H:%M}   |   {out['style']} idea for a {hold_txt} hold")
    if out["tick_info"]:
        st.caption(out["tick_info"])
    if out.get("fallback"):
        st.warning("Live intraday candles are not available right now (market closed or feed empty). This idea uses PREVIOUS DAYS' "
                   "daily chart for a 1-day hold. Confirm at the next market open.")
    for tf, w in out["why"].items():
        st.caption(f"{tf} skipped: {w}")
    t1, t2, t3, t4 = st.tabs(["🎯 Trade idea", "📊 All timeframes", "🕯 Charts and patterns", "📰 News and events"])
    with t1:
        render_idea(o, "single")
        if o["daily"]:
            st.caption(f"Daily trend: {o['daily']}")
    with t2:
        rows = []
        for r in res:
            p, bt = r["plan"], r["bt"]
            rows.append({"TF": r["tf"], "Type": r["kind"], "Direction": ps.dir_name(out["market"], r["side"]), "Score": round(r["score"]),
                         "Entry": ps.fp(p and p["entry"], r["dec"]), "SL": ps.fp(p and p["sl"], r["dec"]), "T1": ps.fp(p and p["t1"], r["dec"]),
                         "T2": ps.fp(p and p["t2"], r["dec"]), "Past T1 hit": f"{bt['win']:.0f}% (n={bt['n']})" if bt and bt["n"] else "-",
                         "Candles": r["nbars"]})
        st.markdown("#### Each timeframe on its own")
        st.dataframe(style_signal(pd.DataFrame(rows)), hide_index=True)
        frows = [{"Horizon": r["fc"]["label"], "TF": r["tf"], "Bias": r["fc"]["bias"], "Chance up": f"{r['fc']['pu']:.0f}%",
                  "Likely range (50%)": f"{ps.fp(r['fc']['lo'], r['dec'])} to {ps.fp(r['fc']['hi'], r['dec'])}",
                  "Median move": f"{r['fc']['med']:+.2f}%", "Basis": r["fc"]["basis"]} for r in res if r["fc"]]
        st.markdown("#### Short-term prediction")
        st.dataframe(pd.DataFrame(frows), hide_index=True)
        st.caption("Range = where price landed half the time after similar setups on this symbol. Not a guarantee.")
        if out["psess"]:
            y = out["psess"][0]
            pos = "above yesterday's high" if price > y["high"] else "below yesterday's low" if price < y["low"] else "inside yesterday's range"
            st.markdown("**Previous sessions:** " + "  |  ".join(f"{p['date']:%d %b}: H {p['high']:,.2f} L {p['low']:,.2f} C {p['close']:,.2f}"
                                                                for p in out["psess"]) + f"  → price is **{pos}**")
    with t3:
        tabs = st.tabs([r["tf"] for r in res])
        for tab, r in zip(tabs, res):
            with tab:
                show_fig(sym, r)
                x = r["d"].iloc[-1]
                st.caption(f"RSI {x.rsi:.0f}  |  ADX {x.adx:.0f}  |  ATR {ps.fp(x.atr, r['dec'])}  |  Support {ps.fp(r['sr']['sup'], r['dec'])}  |  "
                           f"Resistance {ps.fp(r['sr']['res'], r['dec'])}")
                st.write("**Candlestick patterns:** " + (", ".join(r["candles"]) or "none"))
                st.write("**Chart patterns:** " + ("; ".join(f"{n} ({b}) {note}" for n, b, note in r["chart"]) or "none"))
                if r.get("warn"):
                    st.warning(r["warn"])
    with t4:
        render_news(out["nw"])


# ------------------------------------------------------------------ ranked scan
def render_scan(R):
    if R["n"] == 0:
        st.error(f"No symbol had usable data. Likely the market is closed or the data feed returned nothing. Skipped: {R['skipped']}")
        return
    sk = ", ".join(f"{k} {v}" for k, v in R["skipped"].items() if v)
    st.caption(f"{ps.MARKETS[R['market']]}  |  {R['style']} for a {hold_txt} hold  |  scanned {len(R['uni'])}, analysed {R['n']}"
               + (f", skipped: {sk}" if sk else "") + f"  |  data to {R['last_ts']:%d %b %H:%M}  |  {R['secs']:.0f}s")
    if R.get("fallback_n"):
        st.warning(f"Live intraday candles were not available for {R['fallback_n']} symbols (market closed or feed empty"
                   + (f": {R['feed_err']}" if R.get("feed_err") else "") + "). For those, the plan is built from PREVIOUS DAYS' daily chart "
                   "(1-day hold). Confirm at the next market open.")
    t1, t2, t3 = st.tabs(["🏆 Ranked opportunities", "👀 Watchlist", "📰 Market and world news"])
    with t1:
        rk = R["ranked"]
        if not rk:
            st.info("No symbol passed all filters for this holding time. That is normal: most of the time there is no clean setup. "
                    "Check the Watchlist tab, or try a different holding time.")
        else:
            st.markdown("Ranked best first. Rank score blends signal strength, how often similar setups reached the target on that symbol's own history, "
                        "agreement between timeframes, risk checks and news.")
            rows = [{"#": i, "Grade": o["grade"], "Symbol": o["sym"], "Direction": ps.dir_name(o["market"], o["side"]), "Rank": round(o["rank"]),
                     "Win chance": f"{o['bt']['pw'] * 100:.0f}% (n={o['bt']['n']})", "Entry": ps.fp(o["plan"]["entry"], o["dec"]),
                     "SL": ps.fp(o["plan"]["sl"], o["dec"]), "T1": ps.fp(o["plan"]["t1"], o["dec"]), "T2": ps.fp(o["plan"]["t2"], o["dec"]),
                     "Hold to T1": ps.fmt_dur(o["t1_min"], o["market"]),
                     "Why": ", ".join([n for n, p_, _ in o["setups"] if p_ * o["side"] > 0] + [n for n, b, _ in o["pri"]["chart"]
                                                                                           if b == ("bull" if o["side"] > 0 else "bear")])[:48] or "trend + momentum",
                     "News": o.get("news_label", "-")} for i, o in enumerate(rk, 1)]
            st.dataframe(style_signal(pd.DataFrame(rows)), hide_index=True)
            if R["market"] == "PSX":
                st.caption("PSX list shows BUY ideas only.")
            for i, o in enumerate(rk, 1):
                with st.expander(f"#{i}  {o['grade']}  {o['sym']}  {ps.dir_name(o['market'], o['side'])}  {ps.fp(o['px'], o['dec'])}  (rank {o['rank']:.0f})",
                                 expanded=(i == 1)):
                    render_idea(o, f"rk{i}", chart=(i <= 3))
                    if i > 3 and st.checkbox("Show chart", key=f"chart{i}"):
                        show_fig(o["sym"], o["chart_r"])
            csv = pd.DataFrame([dict(rank=i, symbol=o["sym"], direction=ps.dir_name(o["market"], o["side"]), grade=o["grade"], score=round(o["rank"]),
                                     price=o["px"], entry=o["plan"]["entry"], sl=o["plan"]["sl"], t1=o["plan"]["t1"], t2=o["plan"]["t2"],
                                     win_chance_pct=round(o["bt"]["pw"] * 100), timeframe=o["tf"], hold=hold_txt) for i, o in enumerate(rk, 1)])
            st.download_button("Download ranking (CSV)", csv.to_csv(index=False), "ranked_opportunities.csv", "text/csv")
        if R["avoid"]:
            st.markdown("**🔴 Weak, avoid or exit:** " + ", ".join(f"{o['sym']} ({o['final']:+.0f})" for o in R["avoid"]))
    with t2:
        if R["watch"]:
            st.markdown("Close to a setup but something is missing. Wait for confirmation.")
            st.dataframe(pd.DataFrame([{"Symbol": o["sym"], "Lean": ps.dir_name(o["market"], 1 if o["final"] > 0 else -1) if o["final"] else "-",
                                        "Score": round(o["final"]), "Price": ps.fp(o["px"], o["dec"]),
                                        "Missing": "; ".join(o["why_not"] or [lab for lab, ok in o["checks"] if not ok][:3])[:90]} for o in R["watch"]]),
                         hide_index=True)
        else:
            st.info("Nothing on the watchlist right now.")
    with t3:
        st.markdown("#### Market, economy and world headlines")
        render_news(R["ctx"], assets=False)


# ------------------------------------------------------------------ pages
if mode == "Analyse one symbol":
    ex = {"PSX": "OGDC, PPL, HUBC", "Crypto": "BTC, ETH, SOL", "Forex": "EURUSD, GBPUSD, USDJPY, XAUUSD (gold)"}[market]
    sym = st.text_input(f"Symbol (e.g. {ex})", value={"PSX": "OGDC", "Crypto": "BTC", "Forex": "EURUSD"}[market],
                        key=f"sym_{market}").strip().upper()
    name = st.text_input("Company name for better news matching (optional)", value="").strip() if market == "PSX" else ""
    if st.button("Analyse", type="primary"):
        with st.spinner("Downloading data and analysing all timeframes..."):
            try:
                o = ps.analyse_symbol(market, sym, style, hold_min, name or None, use_news, demo)
                o["sig"] = SIG
                st.session_state["single"] = o
            except SystemExit as e:
                st.session_state.pop("single", None)
                st.error(f"Could not load data for {sym}. {e}")
            except Exception as e:
                st.session_state.pop("single", None)
                st.error(f"Could not analyse {sym}: {e}")
    out = st.session_state.get("single")
    if out:
        if out["sig"] == SIG:
            render_single(out)
        else:
            st.info("You changed market, style or holding time. Press **Analyse** again.")
else:
    st.write(f"Scans many {ps.MARKETS[market]} symbols, builds a trade plan sized to your **{hold_txt}** holding time, and ranks the best first.")
    top = st.slider("How many top opportunities to show", 3, 20, 10)
    custom = st.text_input("Only these symbols (optional, space separated)", value="").upper().split()
    min_turn, all_psx, limit = 0, False, 0
    if market == "PSX":
        min_turn = st.number_input("Min turnover last session (PKR million)", min_value=0, value=20, step=5)
        all_psx = st.checkbox("Scan ALL PSX symbols (slow, several minutes)", value=False)
        if all_psx:
            limit = int(st.number_input("Scan only first N symbols (0 = all)", min_value=0, value=0, step=50))
    if st.button("Find best opportunities", type="primary"):
        bar = st.progress(0.0, text="Starting...")

        def cb(done, total, label="Working"):
            bar.progress(min(done / max(total, 1), 1.0), text=f"{label} {done}/{total}")

        try:
            R = ps.scan_best(market, style, hold_min, top, demo, custom, int(min_turn * 1e6), all_psx, limit, use_news, cb)
            R["sig"] = SIG
            st.session_state["scan"] = R
        except SystemExit as e:
            st.error(f"Scan failed. {e}")
        except Exception as e:
            st.error(f"Scan failed: {e}")
        bar.empty()
    R = st.session_state.get("scan")
    if R:
        if R["sig"] == SIG:
            render_scan(R)
        else:
            st.info("You changed market, style or holding time. Press **Find best opportunities** again.")
