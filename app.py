"""
PSX Smart Scanner: web app (works on a phone browser).
Type a stock name -> it downloads PSX data and analyses 5m / 15m / 30m / 1h / 1D.
Run locally:  streamlit run app.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import psx_scanner as ps

st.set_page_config(page_title="PSX Smart Scanner", page_icon="📈", layout="wide")
st.title("📈 PSX Smart Scanner")
st.caption("Educational tool, not financial advice. Always use a stop loss.")

mode = st.sidebar.radio("Mode", ["Single stock", "Market scan (intraday)"])
capital = st.sidebar.number_input("Capital in PKR (for position size)", min_value=0, value=0, step=50000)
risk_pct = st.sidebar.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
demo = st.sidebar.checkbox("Demo data (test only)", value=False)

SIGCOL = {"BUY": "color:#16a34a;font-weight:700", "SELL/EXIT": "color:#dc2626;font-weight:700", "NO TRADE": "color:#ca8a04"}


def fnum(x):
    return "-" if x is None else f"{x:,.2f}"


def style_signal(df, col="Signal"):
    sty = df.style
    fn = getattr(sty, "map", None) or sty.applymap
    return fn(lambda v: SIGCOL.get(v, ""), subset=[col])


def show_fig(sym, r):
    fig = ps.draw(sym, r, fig_only=True)
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ single stock
def render_single(out):
    res, sym, why = out["results"], out["sym"], out["why"]
    price = res[0]["d"].close.iloc[-1]
    d1 = next((r for r in res if r["tf"] == "1D"), None)
    chg = ""
    if d1 is not None and len(d1["d"]) > 1:
        c = d1["d"].close
        chg = f"{100 * (c.iloc[-1] / c.iloc[-2] - 1):+.2f}% last session"
    st.subheader(f"{sym}   PKR {price:,.2f}")
    st.caption(f"{chg}   |   data to {res[0]['d'].index[-1]:%d %b %H:%M}")
    if out["tick_info"]:
        st.caption(out["tick_info"])
    for tf, w in why.items():
        st.warning(f"{tf}: {w}")
    low = [f"{r['tf']} ({r['nbars']})" for r in res if r["nbars"] < 45 and r["tf"] != "1D"]
    if low:
        st.info("Short history for " + ", ".join(low) + ": indicator windows were shortened automatically. Patterns and levels are valid; confidence is lower early in the session.")

    c1, c2 = st.columns(2)
    for col, kind, label in ((c1, "Intraday", "Today (5m + 15m + 30m)"), (c2, "Swing", "1-2 weeks (1h + 1D)")):
        g = ps.group_view(res, kind)
        with col:
            if g:
                st.metric(label, g[1], f"score {g[0]:+.0f}", delta_color="off")
                st.caption(g[3])
            else:
                st.metric(label, "n/a")

    best = ps.best_of(res)
    if best:
        p = best["plan"]
        txt = (f"⭐ **{ps.sig_name(best['side'])} on {best['tf']}** ({best['kind']}), score {best['score']:+.0f}\n\n"
               f"Entry **{p['entry']:,.2f}**  |  SL **{p['sl']:,.2f}** (risk {100 * p['risk'] / p['entry']:.1f}%)  |  "
               f"T1 **{p['t1']:,.2f}** in {best['t1_time']}  |  T2 **{p['t2']:,.2f}** in {best['t2_time']}")
        if capital and best["side"] > 0:
            txt += f"\n\nPosition size: risk {risk_pct}% of {capital:,.0f} = {capital * risk_pct / 100:,.0f} PKR, about **{int(capital * risk_pct / 100 / p['risk'])} shares**"
        (st.success if best["side"] > 0 else st.error)(txt)
    else:
        st.info("No timeframe has a signal that was also profitable in its own history. Waiting is a valid position.")

    rows = []
    for r in res:
        p, bt = r["plan"], r["bt"]
        rows.append({"TF": r["tf"], "Type": r["kind"], "Signal": ps.sig_name(r["side"]), "Score": round(r["score"]),
                     "Entry": fnum(p and p["entry"]), "SL": fnum(p and p["sl"]), "T1": fnum(p and p["t1"]), "T2": fnum(p and p["t2"]),
                     "Time to T1": r.get("t1_time", "-"),
                     "Past T1 hit": f"{bt['win']:.0f}% (n={bt['n']})" if bt and bt["n"] else "-", "Candles": r["nbars"]})
    st.markdown("#### Signals and trade plans")
    st.dataframe(style_signal(pd.DataFrame(rows)), use_container_width=True, hide_index=True)
    st.caption("SELL/EXIT = bearish setup. Retail short selling is usually impractical on PSX: read it as exit or avoid.")

    frows = [{"Horizon": r["fc"]["label"], "TF": r["tf"], "Bias": r["fc"]["bias"], "Chance up": f"{r['fc']['pu']:.0f}%",
              "Likely range (50%)": f"{r['fc']['lo']:,.2f} to {r['fc']['hi']:,.2f}", "Median move": f"{r['fc']['med']:+.2f}%",
              "Basis": r["fc"]["basis"]} for r in res if r["fc"]]
    st.markdown("#### Short-term prediction")
    st.dataframe(pd.DataFrame(frows), use_container_width=True, hide_index=True)
    st.caption("Range = where price landed half the time after similar setups on this stock. Not a guarantee.")

    if out["psess"]:
        y = out["psess"][0]
        pos = "above yesterday's high" if price > y["high"] else "below yesterday's low" if price < y["low"] else "inside yesterday's range"
        st.markdown("**Previous sessions:** " + "  |  ".join(f"{p['date']:%d %b}: H {p['high']:,.2f} L {p['low']:,.2f} C {p['close']:,.2f}" for p in out["psess"]) + f"  → price is **{pos}**")

    nw = out["nw"]
    if nw is not None:
        with st.expander("📰 News check", expanded=True):
            if nw["err"]:
                st.warning(f"News could not be loaded ({nw['err']}).")
            else:
                lab, _ = ps.news_label(nw["score"])
                st.markdown(f"**Stock news mood: {lab}** ({nw['score']:+.0f}). Keyword-based, read the headlines too.")
                for i in nw["stock"] or []:
                    st.write(("🟢 " if i["s"] > 0 else "🔴 " if i["s"] < 0 else "⚪ ") + f"{i['title']}  ·  {i['src']}, {i['age']:.0f}d")
                if not nw["stock"]:
                    st.caption("No stock-specific headlines in the last 7 days.")
                ml, _ = ps.news_label(nw["mscore"])
                st.markdown(f"**Market mood (KSE-100): {ml}** ({nw['mscore']:+.0f})")
                for i in nw["market"][:3]:
                    st.write(("🟢 " if i["s"] > 0 else "🔴 " if i["s"] < 0 else "⚪ ") + f"{i['title']}  ·  {i['src']}")

    st.markdown("#### Charts (each timeframe)")
    tabs = st.tabs([r["tf"] for r in res])
    for tab, r in zip(tabs, res):
        with tab:
            show_fig(sym, r)
            x = r["d"].iloc[-1]
            st.caption(f"RSI {x.rsi:.0f}  |  ADX {x.adx:.0f}  |  ATR {x.atr:.2f}  |  Support {fnum(r['sr']['sup'])}  |  Resistance {fnum(r['sr']['res'])}")
            st.write("**Candles:** " + (", ".join(r["candles"]) or "none"))
            st.write("**Chart patterns:** " + ("; ".join(f"{n} ({b}) {note}" for n, b, note in r["chart"]) or "none"))
            if r.get("warn"):
                st.warning(r["warn"])


# ------------------------------------------------------------------ market scan
def render_scan(R, capital, risk_pct):
    if not R["evs"]:
        st.error(f"No stock had usable intraday data. Likely the market is closed or the feed returned nothing. Skipped: {R['skipped']}")
        return
    evs, uni, buys, watch, bears = R["evs"], R["uni"], R["buys"], R["watch"], R["bears"]
    W, L, T, N = R["pool"]
    st.caption(f"Data to {R['last_ts']:%d %b %Y %H:%M}  |  universe {len(uni)}  |  analysed {len(evs)}  |  skipped "
               + ", ".join(f"{k} {v}" for k, v in R["skipped"].items() if v))
    if R["age"] and R["age"] > 45:
        st.warning(f"Latest tick is {R['age'] / 60:.1f} h old. Market looks closed: these are setups to confirm at the NEXT open, not live entries.")
    if N >= 30:
        exp = (1.5 * W - L) / N
        (st.success if exp > 0.05 else st.warning)(
            f"Same 5m long rule across all scanned stocks (data window only): T1 hit {100 * W / N:.0f}%, SL hit {100 * L / N:.0f}%, avg {exp:+.2f}R over {N} past signals"
            + ("" if exp > 0 else " (weak edge in this window: size down)"))
    st.markdown("### 🟢 Intraday BUY signals")
    if buys:
        rows = [{"Gr": e["grade"], "Stock": e["sym"], "Price": e["px"], "Score": round(e["final"]),
                 "Setup": ", ".join(n for n, p_, _ in e["setups"] if p_ > 0)[:40] or "trend + momentum",
                 "Entry": e["plan"]["entry"], "SL": e["plan"]["sl"], "T1": e["plan"]["t1"], "T2": e["plan"]["t2"],
                 "Hold to T1": ps.fmt_hold(e["mins_t1"]), "RVol": None if np.isnan(e["rv"]) else round(e["rv"], 1), "Daily": e["daily"] or "-"}
                for e in buys]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for k, e in enumerate(buys[:3]):
            with st.expander(f"{e['grade']}  {e['sym']}  {e['px']:,.2f}", expanded=(k == 0)):
                st.write("  ".join(("✅ " if ok else "❌ ") + lab for lab, ok in e["checks"]))
                p, fc = e["plan"], e["r5"]["fc"]
                st.write("**Setups:** " + ("; ".join(f"{n} ({nt})" for n, p_, nt in e["setups"] if p_ > 0) or "none"))
                st.write(f"**Plan:** entry {p['entry']:,.2f}, SL {p['sl']:,.2f} (risk {100 * p['risk'] / p['entry']:.2f}%), "
                         f"T1 {p['t1']:,.2f} in {ps.fmt_hold(e['mins_t1'])}, T2 {p['t2']:,.2f} in {ps.fmt_hold(e['mins_t1'] * 2.8)}")
                st.warning(f"Time-stop: if T1 is not hit within ~{ps.fmt_hold(2 * e['mins_t1'])}, exit. Square off before the market closes.")
                st.write(f"**Next ~1 hour:** {fc['bias']} ({fc['pu']:.0f}% up), range {fc['lo']:,.2f} to {fc['hi']:,.2f}")
                if e["ext"] > 1.8:
                    st.info(f"Extended above EMA: safer entry is a pullback near {e['ema20']:,.2f}.")
                if e["note"]:
                    st.caption("Daily context: " + e["note"])
                if capital:
                    st.write(f"**Size:** about {int(capital * risk_pct / 100 / p['risk'])} shares (risk {risk_pct}% of {capital:,.0f})")
                n_ = R["nw"].get(e["sym"])
                if n_ and not n_["err"] and n_["stock"]:
                    lab, _ = ps.news_label(n_["score"])
                    st.write(f"**News:** {lab} ({n_['score']:+.0f}): {n_['stock'][0]['title'][:90]}")
                r = dict(e["r5"]); r.update(plan=p, side=1, score=e["final"])
                show_fig(e["sym"], r)
    else:
        st.info("No stock passes all 9 checks right now. That is a normal outcome: most of the day has no clean setup. Watch the list below.")
    if watch:
        st.markdown("### 👀 Watchlist (almost there)")
        st.dataframe(pd.DataFrame([{"Stock": e["sym"], "Price": e["px"], "Score": round(e["final"]),
                                    "Trigger": f"close above {e['trig']:,.2f}", "Still needs": ", ".join(e["need"])} for e in watch]),
                     use_container_width=True, hide_index=True)
    if bears:
        st.markdown("**🔴 Weak, avoid or exit:** " + ", ".join(f"{e['sym']} ({e['final']:+.0f})" for e in bears))
    rows = [dict(symbol=e["sym"], type=e["kind"], grade=e["grade"], price=e["px"], score=round(e["final"]), entry=e["plan"]["entry"],
                 sl=e["plan"]["sl"], t1=e["plan"]["t1"], t2=e["plan"]["t2"], hold_to_t1_min=round(e["mins_t1"]),
                 setups="; ".join(n for n, p_, _ in e["setups"] if p_ > 0), need="; ".join(e["need"]), daily=e["daily"]) for e in buys + watch]
    if rows:
        st.download_button("Download signals (CSV)", pd.DataFrame(rows).to_csv(index=False), "psx_signals.csv", "text/csv")


# ------------------------------------------------------------------ pages
if mode == "Single stock":
    sym = st.text_input("Stock symbol (e.g. OGDC, PPL, HUBC)", value="OGDC").strip().upper()
    name = st.text_input("Company name for better news matching (optional)", value="").strip()
    use_news = st.checkbox("Check news", value=True)
    if st.button("Analyse", type="primary"):
        with st.spinner("Downloading data and analysing all timeframes..."):
            try:
                st.session_state["single"] = ps.analyse_stock(sym, demo, name or None, use_news)
            except SystemExit as e:
                st.session_state.pop("single", None)
                st.error(f"Could not load data for {sym}. {e}")
            except Exception as e:
                st.session_state.pop("single", None)
                st.error(f"Could not analyse {sym}: {e}")
    if st.session_state.get("single"):
        render_single(st.session_state["single"])
else:
    st.write("Scans many stocks automatically and lists intraday BUY signals (5m + 15m + 30m, plus daily context).")
    top = st.slider("Max BUY signals", min_value=3, max_value=20, value=10)
    min_turn = st.number_input("Min turnover last session (PKR million)", min_value=0, value=20, step=5)
    limit = st.number_input("Scan only first N stocks (0 = all, slower)", min_value=0, value=0, step=50)
    custom = st.text_input("Or only these symbols (space separated, optional)", value="").upper().split()
    use_news = st.checkbox("Check news for top signals", value=False)
    if st.button("Run scan", type="primary"):
        bar = st.progress(0.0, text="Starting...")

        def cb(done, total):
            bar.progress(min(done / max(total, 1), 1.0), text=f"Downloading {done}/{total}")

        try:
            st.session_state["scan"] = ps.scan_market(custom, demo, top, int(min_turn * 1e6), 6, use_news, int(limit), cb)
        except SystemExit as e:
            st.error(f"Scan failed. {e}")
        except Exception as e:
            st.error(f"Scan failed: {e}")
        bar.empty()
    if st.session_state.get("scan"):
        render_scan(st.session_state["scan"], capital, risk_pct)
