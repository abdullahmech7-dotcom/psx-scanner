#!/usr/bin/env python3
"""
PSX Smart Scanner v2
Type a stock name -> data download -> multi-timeframe technical scan (5m/15m/30m intraday, 1h/1D swing)
-> news check (positive/negative) -> signals, trade plan, short-term forecast, annotated chart.

Terminal / Colab:
    python psx_scanner.py OGDC
    python psx_scanner.py OGDC PPL --capital 500000 --risk-pct 1
    python psx_scanner.py PPL --name "Pakistan Petroleum"      # better news matching
    python psx_scanner.py OGDC --no-news                        # skip news
    python psx_scanner.py OGDC --demo                           # synthetic data, to test
Notebook cell:  scan("OGDC", capital=500000, risk_pct=1)
Educational use only. Not financial advice.
"""
import argparse, os, re, subprocess, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    from rich import box
    from rich.console import Console
    from rich.markup import escape as esc
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rich"])
    from rich import box
    from rich.console import Console
    from rich.markup import escape as esc
    from rich.panel import Panel
    from rich.table import Table

con = Console(force_terminal=True, force_jupyter=False, width=112)
THR = 35            # |score| needed for a BUY / SELL signal
MIN_BARS = {"5m": 10, "15m": 6, "30m": 5, "1h": 4, "1D": 45}   # candles needed per timeframe (intraday: 10 candles is enough)
# name: (kind, backtest horizon in bars, minutes per bar or None)
TFS = {"5m": ("Intraday", 24, 5), "15m": ("Intraday", 16, 15), "30m": ("Intraday", 12, 30),
       "1h": ("Swing", 12, 60), "1D": ("Swing", 10, None)}
# forecast horizon in bars, label
FC = {"5m": (12, "next ~1 hour"), "15m": (8, "next ~2 hours"), "30m": (6, "next ~3 hours"),
      "1h": (6, "next ~1 day (6 hours)"), "1D": (5, "next ~1 week (5 days)")}
INTRADAY_ERR = None
NAMES = {"OGDC": "Oil and Gas Development", "PPL": "Pakistan Petroleum", "HUBC": "Hub Power",
         "ENGRO": "Engro Corporation", "LUCK": "Lucky Cement", "MEBL": "Meezan Bank", "HBL": "Habib Bank",
         "UBL": "United Bank", "MCB": "MCB Bank", "FFC": "Fauji Fertilizer", "PSO": "Pakistan State Oil",
         "SYS": "Systems Limited", "EFERT": "Engro Fertilizers", "POL": "Pakistan Oilfields",
         "MARI": "Mari Petroleum", "TRG": "TRG Pakistan", "NBP": "National Bank of Pakistan",
         "BAFL": "Bank Alfalah", "FCCL": "Fauji Cement", "DGKC": "DG Khan Cement", "MLCF": "Maple Leaf Cement",
         "KEL": "K-Electric", "ATRL": "Attock Refinery", "NRL": "National Refinery", "CNERGY": "Cnergyico",
         "PAEL": "Pak Elektron", "BAHL": "Bank AL Habib", "AIRLINK": "Air Link Communication",
         "SNGP": "Sui Northern Gas", "SSGC": "Sui Southern Gas", "KAPCO": "Kot Addu Power"}


# ------------------------------------------------------------------ data
def _norm(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        col = next((c for c in ("timestamp", "datetime", "time", "date") if c in df.columns), None)
        if col is None:
            try:
                df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
                col = "__index__"
            except Exception:
                raise ValueError(f"No time column found. Columns: {list(df.columns)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            m = s.dropna().abs().max()
            unit = "ns" if m > 1e17 else "us" if m > 1e14 else "ms" if m > 1e11 else "s"
            s = pd.to_datetime(s, unit=unit, utc=True).dt.tz_convert("Asia/Karachi")
        else:
            s = pd.to_datetime(s)
        df = df.drop(columns=[col])
        df.index = pd.DatetimeIndex(s)
    df = df.sort_index()
    return df[~df.index.duplicated(keep="last")]


def _ohlcv(df, allow_synth=False):
    if allow_synth and {"open", "close"} <= set(df.columns):
        df = df.copy()
        if "high" not in df.columns:
            df["high"] = df[["open", "close"]].max(axis=1)
        if "low" not in df.columns:
            df["low"] = df[["open", "close"]].min(axis=1)
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Columns: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]].astype(float).dropna(subset=["close"])


def candles_from(raw, rule):
    """Turn ticks (price, volume) or finer candles into OHLCV candles for `rule`."""
    if all(c in raw.columns for c in ("open", "high", "low", "close")):
        g = _ohlcv(raw).resample(rule)
        out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    else:
        pc = next((c for c in ("price", "close", "last", "rate") if c in raw.columns), None)
        if pc is None:
            raise ValueError(f"No price column in intraday data. Columns: {list(raw.columns)}")
        vc = next((c for c in ("volume", "vol", "qty", "quantity") if c in raw.columns), None)
        p = raw[pc].astype(float)
        out = p.resample(rule).ohlc()
        out["volume"] = raw[vc].astype(float).resample(rule).sum() if vc else 0.0
    return out.dropna(subset=["close"])


def load_live(sym):
    try:
        import pypsx_toolkit as pt
    except ImportError:
        sys.exit("Install first:  pip install pypsx-toolkit")
    t = pt.PSXTicker(sym)
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=730)
    fmt_ = "%Y-%m-%d"
    tries = (lambda: t.get_historical(start_date=start.strftime(fmt_), end_date=end.strftime(fmt_)),
             lambda: t.history(period="2y", interval="1d"))
    daily, partial = None, None
    for fn in tries:
        try:
            cand = _norm(fn())
        except Exception as e:
            print(f"[!] daily source failed: {e}")
            continue
        print("    daily columns:", list(cand.columns))
        if {"open", "high", "low", "close"} <= set(cand.columns):
            daily = _ohlcv(cand)
            break
        partial = partial if partial is not None else cand
    if daily is None:
        if partial is None:
            sys.exit(f"Could not download daily data for {sym}. Check the symbol name.")
        print("[!] This data source has no high/low. Wicks are approximated from open/close:\n"
              "    candlestick patterns and ATR/stop-loss are LESS reliable on 1D. Intraday TFs are unaffected.")
        daily = _ohlcv(partial, allow_synth=True)
    global INTRADAY_ERR
    raw, INTRADAY_ERR = None, None
    try:
        raw = _norm(t.intraday())
    except Exception as e:
        INTRADAY_ERR = str(e)[:160]
        print(f"[!] Intraday data unavailable for {sym}: {e}")
    return daily, raw


def load_hourly_history(sym, days=25):
    """OPTIONAL (needs free pypsx API key in env PYPSX_API_KEY_ID / PYPSX_API_SECRET_KEY): ~25 days of 1h candles."""
    kid, sec = os.environ.get("PYPSX_API_KEY_ID"), os.environ.get("PYPSX_API_SECRET_KEY")
    if not (kid and sec):
        return None
    try:
        from pypsx import TradingClient
        cl = TradingClient(api_key=kid, secret_key=sec, paper=True)
        end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=days)
        rows = cl.get_historical_intraday([sym], start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1h")
        return _ohlcv(_norm(pd.DataFrame(rows)), allow_synth=True)
    except Exception as e:
        print(f"[!] pypsx hourly history failed ({str(e)[:100]}). Using ticks-based 1h candles.")
        return None


def load_demo(sym):
    rng = np.random.default_rng(7)
    n = 520
    r = rng.normal(0.0006, 0.017, n)
    close = 200 * np.exp(np.cumsum(r))
    op = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, .003, n))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, .007, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, .007, n)))
    daily = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close,
                          "volume": rng.integers(5e5, 5e6, n).astype(float)},
                         index=pd.bdate_range(end="2026-11-20", periods=n))
    ticks = []
    price = close[-1]
    for day in pd.bdate_range(end="2026-11-20", periods=3):
        for ts in pd.date_range(day + pd.Timedelta(hours=9, minutes=30),
                                day + pd.Timedelta(hours=15, minutes=25), freq="20s"):
            price *= 1 + rng.normal(0.00002, 0.0007)
            ticks.append((ts, price, int(rng.integers(100, 5000))))
    raw = pd.DataFrame(ticks, columns=["time", "price", "volume"]).set_index("time")
    return daily, raw


# ------------------------------------------------------------------ indicators
def add_ind(d, intraday=False):
    d = d.copy()
    c = d.close
    n = len(d)
    sf, ss, bw_ = min(20, max(4, n // 3)), min(50, max(8, (2 * n) // 3)), min(20, max(5, n // 2))   # windows shrink on short data
    d.attrs["spans"] = (sf, ss)
    d["ema20"] = c.ewm(span=sf, adjust=False).mean()
    d["ema50"] = c.ewm(span=ss, adjust=False).mean()
    dl = c.diff()
    up, dn = dl.clip(lower=0), -dl.clip(upper=0)
    rs = up.ewm(alpha=1 / 14, adjust=False).mean() / dn.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    d["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd"], d["macd_sig"] = macd, macd.ewm(span=9, adjust=False).mean()
    d["mhist"] = d.macd - d.macd_sig
    tr = pd.concat([d.high - d.low, (d.high - c.shift()).abs(), (d.low - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    m, s = c.rolling(bw_).mean(), c.rolling(bw_).std()
    d["bb_mid"], d["bb_up"], d["bb_lo"] = m, m + 2 * s, m - 2 * s
    d["vol_ma"] = d.volume.rolling(bw_, min_periods=3).mean()
    d["hh20"] = d.high.shift().rolling(bw_, min_periods=3).max()
    d["ll20"] = d.low.shift().rolling(bw_, min_periods=3).min()
    # ADX (trend strength)
    u, w_ = d.high.diff(), -d.low.diff()
    plus = pd.Series(np.where((u > w_) & (u > 0), u, 0.0), index=d.index)
    minus = pd.Series(np.where((w_ > u) & (w_ > 0), w_, 0.0), index=d.index)
    pdi = 100 * plus.ewm(alpha=1 / 14, adjust=False).mean() / d.atr
    mdi = 100 * minus.ewm(alpha=1 / 14, adjust=False).mean() / d.atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
    # session VWAP (intraday only)
    d["vwap"] = np.nan
    if intraday and d.volume.sum() > 0:
        tp = (d.high + d.low + d.close) / 3
        day = np.asarray(d.index.date)
        d["vwap"] = (tp * d.volume).groupby(day).cumsum() / d.volume.groupby(day).cumsum().replace(0, np.nan)
    return d


def prev_sessions(daily, ref_date=None, k=2):
    """Last k COMPLETED sessions from daily bars (yesterday, day before). Most recent first."""
    if daily is None or not len(daily):
        return []
    ref = pd.Timestamp(ref_date).date() if ref_date is not None else dt.date.today()
    dd = daily[[i.date() < ref for i in daily.index]].tail(k).iloc[::-1]
    return [dict(date=i.date(), high=float(b.high), low=float(b.low), close=float(b.close)) for i, b in dd.iterrows()]


def candle_patterns(d):
    o, h, l, c = d.open, d.high, d.low, d.close
    body, rng = (c - o).abs(), (h - l)
    up_w, lo_w = h - np.maximum(o, c), np.minimum(o, c) - l
    bull, bear = c > o, c < o
    po, pc_, pbody = o.shift(), c.shift(), body.shift()
    down_ctx, up_ctx = c < d.ema20, c > d.ema20
    big = body > body.rolling(10).mean()
    P = {
        "Hammer": (lo_w >= 2 * body) & (up_w <= 0.3 * body + 1e-9) & (body > 0) & down_ctx,
        "Inverted hammer": (up_w >= 2 * body) & (lo_w <= 0.3 * body + 1e-9) & (body > 0) & down_ctx,
        "Shooting star": (up_w >= 2 * body) & (lo_w <= 0.3 * body + 1e-9) & (body > 0) & up_ctx,
        "Bullish engulfing": bear.shift() & bull & (o <= pc_) & (c >= po),
        "Bearish engulfing": bull.shift() & bear & (o >= pc_) & (c <= po),
        "Morning star": bear.shift(2) & (body.shift() < 0.3 * body.shift(2)) & bull & (c > (o.shift(2) + c.shift(2)) / 2),
        "Evening star": bull.shift(2) & (body.shift() < 0.3 * body.shift(2)) & bear & (c < (o.shift(2) + c.shift(2)) / 2),
        "Bullish harami": bear.shift() & (pbody > body.rolling(10).mean().shift()) & bull & (o > pc_) & (c < po),
        "Bearish harami": bull.shift() & (pbody > body.rolling(10).mean().shift()) & bear & (o < pc_) & (c > po),
        "Bullish marubozu": bull & big & (body >= 0.9 * rng),
        "Bearish marubozu": bear & big & (body >= 0.9 * rng),
        "Doji": (body <= 0.1 * rng) & (rng > 0),
    }
    P = {k: v.fillna(False).astype(bool) for k, v in P.items()}
    bn = ["Hammer", "Inverted hammer", "Bullish engulfing", "Morning star", "Bullish harami", "Bullish marubozu"]
    brn = ["Shooting star", "Bearish engulfing", "Evening star", "Bearish harami", "Bearish marubozu"]
    return P, bn, brn


def score_series(d, P, bn, brn):
    c = d.close
    s = (np.where(c > d.ema20, 20, -20) + np.where(d.ema20 > d.ema50, 15, -15)
         + np.sign(d.mhist) * 10 + np.sign(d.mhist.diff()) * 5
         + (d.rsi - 50) * 0.3 + np.where(d.rsi > 75, -10, 0) + np.where(d.rsi < 25, 10, 0)
         + np.where(c > d.hh20, 20, 0) + np.where(c < d.ll20, -20, 0))
    cb = pd.concat([P[k] for k in bn], axis=1).any(axis=1).astype(int).rolling(2, min_periods=1).max()
    cr = pd.concat([P[k] for k in brn], axis=1).any(axis=1).astype(int).rolling(2, min_periods=1).max()
    s = s + 10 * cb - 10 * cr + np.where(d.volume > 1.5 * d.vol_ma, 5 * np.sign(c - d.open), 0)
    s = s + np.sign(c - d.vwap).fillna(0) * 10            # above/below session VWAP (intraday)
    s = s * np.where(d.adx < 18, 0.75, 1.0)               # weak trend / range market -> less conviction
    return pd.Series(s, index=d.index).fillna(0).clip(-100, 100)


# ------------------------------------------------------------------ levels + chart patterns
def pivots(d, k=3):
    h, l = d.high.values, d.low.values
    ph, pl = [], []
    for i in range(k, len(d) - k):
        if h[i] == h[i - k:i + k + 1].max():
            ph.append(i)
        if l[i] == l[i - k:i + k + 1].min():
            pl.append(i)
    return ph, pl


def sr_levels(d):
    w = d.iloc[-150:]
    a, c = w.atr.iloc[-1], w.close.iloc[-1]
    ph, pl = pivots(w)
    pts = sorted([w.high.iloc[i] for i in ph] + [w.low.iloc[i] for i in pl])
    cl = []
    for p in pts:
        if cl and abs(p - np.mean(cl[-1])) <= 0.6 * a:
            cl[-1].append(p)
        else:
            cl.append([p])
    lv = [(float(np.mean(x)), len(x)) for x in cl if len(x) >= 2] or [(float(np.mean(x)), len(x)) for x in cl]
    sup = max([x for x, _ in lv if x < c], default=None)
    res = min([x for x, _ in lv if x > c], default=None)
    return dict(levels=sorted(lv, key=lambda t: abs(t[0] - c))[:5], sup=sup, res=res)


def chart_patterns(d):
    w = d.iloc[-120:]
    h, l, c, a = w.high.values, w.low.values, w.close.iloc[-1], w.atr.iloc[-1]
    n = len(w)
    ph, pl = pivots(w)
    out = []
    if len(ph) >= 2 and n - 1 - ph[-1] <= 25 and abs(h[ph[-1]] - h[ph[-2]]) <= a and ph[-1] - ph[-2] >= 5:
        tr = l[ph[-2]:ph[-1]].min()
        if h[ph[-1]] - tr >= 2 * a:
            out.append(("Double top", "bear", "CONFIRMED, broke %.2f" % tr if c < tr else "forming, neckline %.2f" % tr))
    if len(pl) >= 2 and n - 1 - pl[-1] <= 25 and abs(l[pl[-1]] - l[pl[-2]]) <= a and pl[-1] - pl[-2] >= 5:
        pk = h[pl[-2]:pl[-1]].max()
        if pk - l[pl[-1]] >= 2 * a:
            out.append(("Double bottom", "bull", "CONFIRMED, broke %.2f" % pk if c > pk else "forming, neckline %.2f" % pk))
    if len(ph) >= 2 and len(pl) >= 2 and n - 1 - min(ph[-1], pl[-1]) <= 30:
        flat_top = abs(h[ph[-1]] - h[ph[-2]]) <= 0.8 * a
        flat_bot = abs(l[pl[-1]] - l[pl[-2]]) <= 0.8 * a
        rising = l[pl[-1]] > l[pl[-2]] + 0.3 * a
        falling = h[ph[-1]] < h[ph[-2]] - 0.3 * a
        if flat_top and rising:
            out.append(("Ascending triangle", "bull", "flat top %.2f, rising lows" % h[ph[-1]]))
        elif flat_bot and falling:
            out.append(("Descending triangle", "bear", "flat base %.2f, falling highs" % l[pl[-1]]))
        elif rising and falling:
            out.append(("Symmetrical triangle", "neutral", "coiling, breakout pending"))
    if c > w.hh20.iloc[-1]:
        out.append(("20-bar breakout", "bull", "closed above %.2f" % w.hh20.iloc[-1]))
    if c < w.ll20.iloc[-1]:
        out.append(("20-bar breakdown", "bear", "closed below %.2f" % w.ll20.iloc[-1]))
    bw = (w.bb_up - w.bb_lo) / w.bb_mid
    if bw.notna().sum() > 40 and bw.iloc[-1] <= bw.dropna().iloc[-100:].quantile(0.10):
        out.append(("Bollinger squeeze", "neutral", "volatility very low, big move soon"))
    return out


# ------------------------------------------------------------------ plan, backtest, forecast
def make_plan(d, i, side):
    e, a = d.close.iloc[i], d.atr.iloc[i]
    lo, hi = d.low.iloc[max(0, i - 9):i + 1].min(), d.high.iloc[max(0, i - 9):i + 1].max()
    if side > 0:
        sl = min(max(lo, e - 2.5 * a), e - 1.0 * a); r = e - sl
        return dict(entry=e, sl=sl, t1=e + 1.5 * r, t2=e + 2.5 * r, risk=r, atr=a)
    sl = max(min(hi, e + 2.5 * a), e + 1.0 * a); r = sl - e
    return dict(entry=e, sl=sl, t1=e - 1.5 * r, t2=e - 2.5 * r, risk=r, atr=a)


def backtest(d, sc, side, H):
    idx = np.where(sc.values >= THR)[0] if side > 0 else np.where(sc.values <= -THR)[0]
    prev = set(idx)
    hi, lo = d.high.values, d.low.values
    w = l = t = 0
    for i in idx:
        if i < 30 or (i - 1) in prev or i + H >= len(d):   # warm-up, streak starts only, full horizon only
            continue
        p = make_plan(d, i, side)
        res = "t"
        for j in range(i + 1, i + 1 + H):
            if side > 0:
                if lo[j] <= p["sl"]: res = "l"; break      # SL first if both hit (conservative)
                if hi[j] >= p["t1"]: res = "w"; break
            else:
                if hi[j] >= p["sl"]: res = "l"; break
                if lo[j] <= p["t1"]: res = "w"; break
        w += res == "w"; l += res == "l"; t += res == "t"
    n = w + l + t
    if n == 0:
        return dict(n=0, win=None, loss=None, exp=None)
    return dict(n=n, win=100 * w / n, loss=100 * l / n, exp=(1.5 * w - l) / n)


def est_time(tf, plan, target):
    a = plan["atr"]
    bars = max(1.0, (abs(target - plan["entry"]) / (0.7 * a)) ** 2)
    mins = TFS[tf][2]
    if mins:
        total = bars * mins
        return f"~{bars:.0f} bars (~{total:.0f} min)" if total <= 360 else f"~{bars:.0f} bars (~{total / 360:.1f} days)"
    return f"~{bars:.0f} days"


def forecast(tf, d, sc):
    """Short-term outlook: what happened after similar scores on this stock; ATR fallback if sample is small."""
    H, label = FC[tf]
    s = float(sc.iloc[-1])
    last, a = d.close.iloc[-1], d.atr.iloc[-1]
    b = 1 if s >= 15 else -1 if s <= -15 else 0
    bucket = (sc >= 15) if b == 1 else (sc <= -15) if b == -1 else (sc.abs() < 15)
    fwd = d.close.shift(-H) / d.close - 1
    m = bucket & fwd.notna()
    m.iloc[:30] = False
    r = fwd[m]
    if len(r) >= 25:
        q1, med, q3 = r.quantile([.25, .5, .75])
        pu, lo, hi, basis = 100 * (r > 0).mean(), last * (1 + q1), last * (1 + q3), f"history n={len(r)}"
    else:
        sig = 0.7 * a * np.sqrt(H)
        tilt = np.clip(s / 100, -1, 1) * 0.35 * sig
        pu, med = 50 + 25 * np.clip(s / 100, -1, 1), tilt / last
        lo, hi, basis = last + tilt - 0.67 * sig, last + tilt + 0.67 * sig, "ATR-based (small history)"
    bias = "UP" if pu >= 57 else "DOWN" if pu <= 43 else "SIDEWAYS"
    return dict(H=H, label=label, pu=float(pu), med=float(med) * 100, lo=float(lo), hi=float(hi), bias=bias, basis=basis)


# ------------------------------------------------------------------ analysis
def analyse(tf, df, extra_levels=None):
    kind = TFS[tf][0]
    d = add_ind(df, intraday=(kind == "Intraday"))
    P, bn, brn = candle_patterns(d)
    sc = score_series(d, P, bn, brn)
    s = float(sc.iloc[-1])
    side = 1 if s >= THR else -1 if s <= -THR else 0
    last = len(d) - 1
    recent = [k + ("" if j == 0 else f" ({j} bar ago)") for j in range(3) for k, v in P.items()
              if last - j >= 0 and v.iloc[last - j] and (k != "Doji" or j == 0)]
    res = dict(tf=tf, kind=kind, d=d, sc=sc, nbars=len(d), score=s, side=side, candles=recent[:3], chart=chart_patterns(d),
               sr=sr_levels(d), fc=forecast(tf, d, sc) if tf in FC else None, plan=None, bt=None,
               spans=d.attrs.get("spans", (20, 50)))
    if extra_levels:                                           # yesterday / day-before highs and lows
        px_, sr = d.close.iloc[-1], res["sr"]
        sr["levels"] = sorted(sr["levels"] + [(float(v), 3) for v in extra_levels], key=lambda t: abs(t[0] - px_))[:8]
        lv_ = [x for x, _ in sr["levels"]]
        sr["sup"] = max([x for x in lv_ if x < px_], default=None)
        sr["res"] = min([x for x in lv_ if x > px_], default=None)
    if side:
        p = make_plan(d, last, side)
        res["plan"], res["bt"] = p, backtest(d, sc, side, TFS[tf][1])
        res["t1_time"], res["t2_time"] = est_time(tf, p, p["t1"]), est_time(tf, p, p["t2"])
        sr = res["sr"]
        if side > 0 and sr["res"] and p["entry"] < sr["res"] < p["t1"]:
            res["warn"] = f"resistance {sr['res']:.2f} sits before Target 1"
        if side < 0 and sr["sup"] and p["t1"] < sr["sup"] < p["entry"]:
            res["warn"] = f"support {sr['sup']:.2f} sits before Target 1"
        mins = TFS[tf][2]
        if mins and kind == "Intraday":
            tm = max(1.0, (abs(p["t1"] - p["entry"]) / (0.7 * p["atr"])) ** 2) * mins
            if tm > 360:
                msg = f"T1 needs ~{tm / 360:.1f} sessions: too slow for same-day, treat as swing"
                res["warn"] = (res["warn"] + "; " if res.get("warn") else "") + msg
                res["slow"] = True
    return res


# ------------------------------------------------------------------ news
POS = set("""profit profits surge surges rally rallies gain gains jump jumps rise rises rose record growth dividend upgrade
upgrades bullish beat beats strong higher boost boosts discovery approval approves buyback expansion wins improves
improvement recovery rebound soars climbs payout bonus outperform increase increases upbeat""".split())
NEG = set("""loss losses fall falls fell drop drops plunge plunges slump decline declines downgrade downgrades bearish weak
lower default probe penalty fine ban lawsuit strike shutdown delay delays crisis slashes concern concerns warns
warning tumble tumbles selloff sell-off halt halts suspend suspended circular shortfall misses decrease decreases
fraud scam plummets""".split())


def head_score(title):
    words = re.findall(r"[a-z\-]+", title.lower())
    return int(np.clip(sum(w in POS for w in words) - sum(w in NEG for w in words), -2, 2))


def _rss(q, limit=20):
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(q) + "&hl=en-PK&gl=PK&ceid=PK:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=12).read())
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for it in list(root.iter("item"))[:limit]:
        title, src = (it.findtext("title") or "").strip(), (it.findtext("source") or "").strip()
        if src and title.endswith(" - " + src):
            title = title[: -(len(src) + 3)]
        try:
            age = max(0.0, (now - parsedate_to_datetime(it.findtext("pubDate"))).total_seconds() / 86400)
        except Exception:
            age = 7.0
        out.append(dict(title=title, src=src or "news", age=age, s=head_score(title)))
    return out


def _agg(items):
    if not items:
        return 0.0
    w = [1.0 if i["age"] <= 2 else 0.7 if i["age"] <= 4 else 0.4 for i in items]
    return float(np.clip(50 * sum(wi * i["s"] for wi, i in zip(w, items)) / sum(w), -100, 100))


def get_news(sym, name=None):
    name = name or NAMES.get(sym)
    try:
        qs = [f'"{name}" when:7d'] if name else []
        qs += [f'"{sym}" PSX when:7d']
        seen, items = set(), []
        for q in qs:
            for it in _rss(q):
                if it["title"] not in seen:
                    seen.add(it["title"]); items.append(it)
        pat = re.compile(r"\b" + re.escape(sym) + r"\b", re.I)
        keys = [w for w in (name or "").lower().split()[:2]]
        stock = [i for i in items if pat.search(i["title"]) or (keys and all(k in i["title"].lower() for k in keys))]
        mkt = _rss('KSE-100 OR "Pakistan Stock Exchange" when:3d', 8)
        stock.sort(key=lambda i: i["age"])
        return dict(stock=stock[:6], market=mkt[:4], score=_agg(stock), mscore=_agg(mkt), err=None)
    except Exception as e:
        return dict(stock=[], market=[], score=0.0, mscore=0.0, err=str(e)[:120])


def demo_news():
    S = lambda t, a: dict(title=t, src="Demo Wire", age=a, s=head_score(t))
    st = [S("Company reports record profit and announces higher dividend", 0.5),
          S("Brokerage upgrades target price on strong growth outlook", 1.2),
          S("Regulator probe raises concern over delay in project", 3.0)]
    mk = [S("KSE-100 rallies as inflation eases", 0.3), S("Index falls on profit taking", 1.0)]
    return dict(stock=st, market=mk, score=_agg(st), mscore=_agg(mk), err=None)


def news_label(x):
    return ("POSITIVE", "green") if x >= 25 else ("NEGATIVE", "red") if x <= -25 else ("MIXED / NEUTRAL", "yellow")


# ------------------------------------------------------------------ report
SIG = {"BUY": "bold green", "SELL/EXIT": "bold red", "NO TRADE": "yellow"}
BIAS = {"UP": "green", "DOWN": "red", "SIDEWAYS": "yellow"}


def sig_name(side):
    return "BUY" if side > 0 else "SELL/EXIT" if side < 0 else "NO TRADE"


def bar(p, n=10):
    k = int(round(p / 100 * n))
    return "█" * k + "░" * (n - k)


def fnum(x):
    return "-" if x is None else f"{x:,.2f}"


def group_view(results, kind):
    g = [r for r in results if r["kind"] == kind]
    if not g:
        return None
    sc = float(np.mean([r["score"] for r in g]))
    lab, col = ("BULLISH", "green") if sc >= 30 else ("BEARISH", "red") if sc <= -30 else ("MIXED / NEUTRAL", "yellow")
    return sc, lab, col, "  ".join(f"{r['tf']} {r['score']:+.0f}" for r in g)


def best_of(results):
    c = [r for r in results if r["side"] and not r.get("slow") and r["bt"] and r["bt"]["n"] >= 10 and r["bt"]["exp"] > 0]
    c.sort(key=lambda r: (r["bt"]["exp"], abs(r["score"])), reverse=True)
    return c[0] if c else None


def report(sym, results, news, capital, risk_pct, why=None, psess=None):
    by = {r["tf"]: r for r in results}
    why = why or {}
    price = results[0]["d"].close.iloc[-1]
    chg = ""
    if "1D" in by and len(by["1D"]["d"]) > 1:
        dd = by["1D"]["d"].close
        ch = 100 * (dd.iloc[-1] / dd.iloc[-2] - 1)
        chg = f"   [{'green' if ch >= 0 else 'red'}]{ch:+.2f}% last session[/]"
    con.print()
    con.print(Panel(f"[bold white]{esc(sym)}[/]   PKR [bold]{price:,.2f}[/]{chg}\n"
                    f"[dim]data as of {results[0]['d'].index[-1]}[/]",
                    title="📈 PSX SMART SCAN", border_style="cyan", box=box.ROUNDED))

    # --- data warnings (never silently skip a timeframe)
    intra_missing = [tf for tf in ("5m", "15m", "30m") if tf in why]
    if len(intra_missing) == 3:
        hard = bool(INTRADAY_ERR) or all("no intraday" in why[t] for t in intra_missing)
        con.print(Panel("\n".join(f"[yellow]{t}[/]: {esc(why[t])}" for t in intra_missing) +
                        ("\n[dim]The feed returned nothing/an error. Re-run in market hours; if it repeats, paste this message to me.[/]"
                         if hard else "\n[dim]Only a few candles exist so far today. Wait a few minutes, or use the daily and previous-session levels below.[/]"),
                        title="⚠ " + ("INTRADAY DATA MISSING" if hard else "NOT ENOUGH INTRADAY CANDLES YET"), border_style="red"))
    elif why:
        con.print(Panel("\n".join(f"[yellow]{tf}[/]: {esc(w)}" for tf, w in why.items()), title="⚠ NOT ANALYSED",
                        border_style="yellow"))
    low = [f"{r['tf']} ({r['nbars']} candles)" for r in results if r["nbars"] < 45 and r["tf"] != "1D"]
    if low:
        con.print(f"[yellow]Short history: {', '.join(low)}. Indicator windows were shortened automatically. Patterns and "
                  f"levels are still valid; trend filters are rougher. Confidence is lower early in the session.[/]")

    # --- verdict
    v = Table(box=box.SIMPLE_HEAD, title="VERDICT", title_style="bold cyan", expand=True)
    for c in ("Horizon", "View", "Score", "Timeframes", "Note"):
        v.add_column(c)
    d1 = group_view(results, "Swing")
    for kind, tag in (("Intraday", "Today (5m + 15m + 30m)"), ("Swing", "1-2 weeks (swing: 1h + 1D)")):
        g = group_view(results, kind)
        if not g:
            continue
        sc, lab, col, det = g
        note = ""
        if kind == "Intraday" and d1 and abs(sc) >= 30 and abs(d1[0]) >= 30 and np.sign(sc) != np.sign(d1[0]):
            note = "[yellow]counter-trend vs swing view, lower confidence[/]"
        elif kind == "Intraday" and d1 and abs(sc) >= 30 and np.sign(sc) == np.sign(d1[0]):
            note = "[green]aligned with swing view[/]"
        v.add_row(tag, f"[bold {col}]{lab}[/]", f"{sc:+.0f}", det, note)
    con.print(v)

    # --- signals + plans
    t = Table(box=box.SIMPLE_HEAD, title="SIGNALS & TRADE PLANS", title_style="bold cyan", expand=True)
    for c in ("TF", "Type", "Signal", "Score", "Entry", "SL", "T1", "T2", "Time to T1", "Past T1-hit"):
        t.add_column(c, justify="right" if c not in ("TF", "Type", "Signal") else "left")
    for r in results:
        p, bt = r["plan"], r["bt"]
        hit = "-"
        if bt and bt["n"]:
            hit = f"{bt['win']:.0f}% (n={bt['n']})" + ("" if bt["n"] >= 15 else " [dim]low[/]")
        t.add_row(r["tf"], r["kind"], f"[{SIG[sig_name(r['side'])]}]{sig_name(r['side'])}[/]", f"{r['score']:+.0f}",
                  fnum(p and p["entry"]), fnum(p and p["sl"]), fnum(p and p["t1"]), fnum(p and p["t2"]),
                  r.get("t1_time", "-"), hit)
    con.print(t)
    con.print("[dim]SELL/EXIT = bearish setup. Retail short selling is usually impractical on PSX: read it as exit / avoid.[/]")

    # --- forecast
    f = Table(box=box.SIMPLE_HEAD, title="SHORT-TERM PREDICTION", title_style="bold cyan", expand=True)
    for c in ("Horizon", "TF", "Bias", "Chance up", "Likely range (50%)", "Median move", "Basis"):
        f.add_column(c)
    for r in results:
        x = r["fc"]
        if not x:
            continue
        col = BIAS[x["bias"]]
        f.add_row(x["label"], r["tf"], f"[bold {col}]{x['bias']}[/]", f"[{col}]{bar(x['pu'])}[/] {x['pu']:.0f}%",
                  f"{x['lo']:,.2f} to {x['hi']:,.2f}", f"{x['med']:+.2f}%", f"[dim]{x['basis']}[/]")
    con.print(f)
    con.print("[dim]Range = where price landed half the time after similar setups on this stock. Not a guarantee.[/]")

    # --- news
    if news is not None:
        lab, col = news_label(news["score"])
        body = []
        if news["err"]:
            body.append(f"[yellow]News could not be loaded ({esc(news['err'])}). Check internet and retry.[/]")
        else:
            if news["stock"]:
                body.append(f"[bold]Stock news mood:[/] [bold {col}]{lab}[/]  ({news['score']:+.0f})  [dim]keyword-based, read headlines too[/]")
                for i in news["stock"]:
                    ic = "🟢" if i["s"] > 0 else "🔴" if i["s"] < 0 else "⚪"
                    body.append(f" {ic} {esc(i['title'][:88])} [dim]{esc(i['src'])}, {i['age']:.0f}d[/]")
            else:
                body.append("[dim]No stock-specific headlines found in the last 7 days.[/]")
            ml, mc = news_label(news["mscore"])
            body.append(f"\n[bold]Market mood (KSE-100 headlines):[/] [bold {mc}]{ml}[/]  ({news['mscore']:+.0f})")
            for i in news["market"][:3]:
                ic = "🟢" if i["s"] > 0 else "🔴" if i["s"] < 0 else "⚪"
                body.append(f" {ic} {esc(i['title'][:88])} [dim]{esc(i['src'])}, {i['age']:.0f}d[/]")
        con.print(Panel("\n".join(body), title="📰 NEWS CHECK", border_style="magenta", box=box.ROUNDED))

    # --- details
    dt_ = Table(box=box.SIMPLE_HEAD, title="WHAT THE CHARTS SHOW", title_style="bold cyan", expand=True)
    for c in ("TF", "Trend", "RSI", "ADX", "MACD", "Vol x", "Support", "Resist"):
        dt_.add_column(c)
    for r in results:
        x = r["d"].iloc[-1]
        trend = "[green]Up[/]" if x.close > x.ema20 > x.ema50 else "[red]Down[/]" if x.close < x.ema20 < x.ema50 else "[yellow]Mixed[/]"
        adx = f"{x.adx:.0f}" + (" [dim]weak[/]" if x.adx < 18 else " [green]strong[/]" if x.adx > 25 else "")
        vr = x.volume / x.vol_ma if x.vol_ma and x.vol_ma > 0 else float("nan")
        dt_.add_row(r["tf"], trend, f"{x.rsi:.0f}", adx, "[green]+[/]" if x.mhist > 0 else "[red]-[/]", f"{vr:.1f}",
                    fnum(r["sr"]["sup"]), fnum(r["sr"]["res"]))
    con.print(dt_)
    if psess:
        pxn, y = results[0]["d"].close.iloc[-1], psess[0]
        pos = "above yesterday's high" if pxn > y["high"] else "below yesterday's low" if pxn < y["low"] else "inside yesterday's range"
        con.print("[bold]Previous sessions:[/] " + "   |   ".join(f"{p['date']:%d %b}: H {p['high']:,.2f}  L {p['low']:,.2f}  C {p['close']:,.2f}" for p in psess)
                  + f"   -> price is [bold]{pos}[/]")
    for r in results:
        pat = "; ".join(f"{esc(n)} ({b})" + f" {esc(note)}" for n, b, note in r["chart"]) or "none"
        con.print(f"[bold]{r['tf']:>3}[/] candles: {esc(', '.join(r['candles']) or 'none')}   |   patterns: {pat}")
        if r.get("warn"):
            con.print(f"     [yellow]warning: {esc(r['warn'])}[/]")

    # --- best opportunity
    b_ = best_of(results)
    cands = [b_] if b_ else []
    if cands:
        b = cands[0]
        p, bt = b["plan"], b["bt"]
        lines = [f"[bold {('green' if b['side'] > 0 else 'red')}]{sig_name(b['side'])}[/] on [bold]{b['tf']}[/] ({b['kind']})   score {b['score']:+.0f}",
                 f"Entry {p['entry']:,.2f}   SL {p['sl']:,.2f} (risk {100 * p['risk'] / p['entry']:.1f}%)   "
                 f"T1 {p['t1']:,.2f} in {b['t1_time']}   T2 {p['t2']:,.2f} in {b['t2_time']}",
                 f"Past on this stock: T1 hit {bt['win']:.0f}%, SL hit {bt['loss']:.0f}%, avg {bt['exp']:+.2f}R, n={bt['n']}"]
        if capital and b["side"] > 0:
            lines.append(f"Position size: risk {risk_pct}% of {capital:,.0f} = {capital * risk_pct / 100:,.0f} PKR -> about "
                         f"{int(capital * risk_pct / 100 / p['risk'])} shares")
        if news and not news["err"] and news["stock"]:
            if b["side"] > 0 and news["score"] <= -25:
                lines.append("[yellow]News is negative: conflicts with the BUY. Wait for confirmation or cut size.[/]")
            elif b["side"] < 0 and news["score"] >= 25:
                lines.append("[yellow]News is positive: conflicts with the bearish setup. Be careful.[/]")
            elif abs(news["score"]) >= 25:
                lines.append("[green]News direction agrees with the setup.[/]")
        if b.get("warn"):
            lines.append(f"[yellow]{esc(b['warn'])}[/]")
        con.print(Panel("\n".join(lines), title="⭐ BEST OPPORTUNITY", border_style="green", box=box.ROUNDED))
    else:
        con.print(Panel("No timeframe has a signal that was also profitable in its own history.\n[bold]Waiting is a valid position.[/]",
                        title="⭐ BEST OPPORTUNITY", border_style="yellow", box=box.ROUNDED))
    con.print("[dim]Educational use only. Not financial advice. Always use a stop loss. Overlays and forecasts are statistical estimates.[/]\n")
    return cands[0] if cands else None


# ------------------------------------------------------------------ chart
def draw(sym, r, show=True, fig_only=False):
    import matplotlib
    if not show or fig_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    BG, FG, UP, DN = "#0f1218", "#d7dbe3", "#22c55e", "#ef4444"
    d, p, full = r["d"].iloc[-90:], r["plan"], r["d"]
    n, x = len(d), np.arange(len(d))
    fig = plt.figure(figsize=(13, 8.2), facecolor=BG)
    gs = gridspec.GridSpec(3, 1, height_ratios=[5, 1.3, 1.3], hspace=.06)
    ax = fig.add_subplot(gs[0]); ar = fig.add_subplot(gs[1], sharex=ax); am = fig.add_subplot(gs[2], sharex=ax)
    for a in (ax, ar, am):
        a.set_facecolor(BG); a.tick_params(colors=FG, labelsize=8); a.grid(alpha=.12)
        for sp in a.spines.values():
            sp.set_color("#2a303b")
    for i, (o, h, l, c) in enumerate(zip(d.open, d.high, d.low, d.close)):
        col = UP if c >= o else DN
        ax.vlines(i, l, h, color=col, lw=1)
        ax.add_patch(plt.Rectangle((i - .32, min(o, c)), .64, max(abs(c - o), 1e-9), color=col))
    sp = r.get("spans", (20, 50))
    ax.plot(x, d.ema20, color="#60a5fa", lw=1.3, label=f"EMA {sp[0]}")
    ax.plot(x, d.ema50, color="#c084fc", lw=1.3, label=f"EMA {sp[1]}")
    ax.fill_between(x, d.bb_lo, d.bb_up, color="#94a3b8", alpha=.10, label="Bollinger")
    if d.vwap.notna().any():
        ax.plot(x, d.vwap, color="#fbbf24", lw=1.1, ls=":", label="VWAP")
    for lv, k in r["sr"]["levels"]:
        if d.low.min() * .98 <= lv <= d.high.max() * 1.02:
            ax.axhline(lv, color="#64748b", lw=.9, ls="--", alpha=.8)
            ax.text(0, lv, f" S/R {lv:.2f}", color="#94a3b8", fontsize=7, va="bottom")
    shown = 0
    for k, v in candle_patterns(full)[0].items():
        if k == "Doji":
            continue
        for j in np.where(v.iloc[-90:].values)[0]:
            if j < n - 12 or shown >= 5:
                continue
            ax.annotate(k, (j, d.high.iloc[j]), xytext=(0, 14 + 12 * (shown % 3)), textcoords="offset points", ha="center",
                        fontsize=8, color="#fbbf24", arrowprops=dict(arrowstyle="-", color="#fbbf24"))
            shown += 1
    if p:
        x0, w = n + 1, 8
        ax.add_patch(plt.Rectangle((x0, min(p["entry"], p["t2"])), w, abs(p["t2"] - p["entry"]), color=UP, alpha=.25))
        ax.add_patch(plt.Rectangle((x0, min(p["entry"], p["sl"])), w, abs(p["sl"] - p["entry"]), color=DN, alpha=.25))
        for lab, v, col in (("Entry", p["entry"], "#60a5fa"), ("SL", p["sl"], "#f87171"), ("T1", p["t1"], "#4ade80"), ("T2", p["t2"], "#22c55e")):
            ax.hlines(v, n - 1, x0 + w, colors=col, linestyles="--", lw=1.2)
            ax.text(x0 + w + .5, v, f"{lab} {v:.2f}", color=col, va="center", fontsize=9, fontweight="bold")
    fc = r["fc"]
    if fc:
        ax.vlines(n + 20, fc["lo"], fc["hi"], color="#fbbf24", lw=7, alpha=.75)
        ax.text(n + 20.8, (fc["lo"] + fc["hi"]) / 2, f"Forecast\n{fc['label']}\n{fc['lo']:.2f}-{fc['hi']:.2f}", color="#fbbf24", fontsize=7, va="center")
    ax.set_xlim(-1, n + 32)
    ax.legend(loc="upper left", fontsize=8, facecolor=BG, labelcolor=FG, framealpha=.6)
    ax.set_title(f"{sym}  {r['tf']}   |   {sig_name(r['side'])}   |   score {r['score']:+.0f}", color=FG, fontsize=13, fontweight="bold")
    ar.plot(x, d.rsi, color="#60a5fa", lw=1.2); ar.axhline(70, color=DN, lw=.7, ls="--"); ar.axhline(30, color=UP, lw=.7, ls="--")
    ar.set_ylim(0, 100); ar.set_ylabel("RSI", color=FG, fontsize=8)
    am.bar(x, d.mhist, color=[UP if v >= 0 else DN for v in d.mhist], width=.7)
    am.plot(x, d.macd, color="#60a5fa", lw=1); am.plot(x, d.macd_sig, color="#fbbf24", lw=1); am.set_ylabel("MACD", color=FG, fontsize=8)
    ticks = np.linspace(0, n - 1, 7).astype(int)
    fmt_ = "%d %b %y" if r["tf"] == "1D" else "%d %b %H:%M"
    am.set_xticks(ticks); am.set_xticklabels([d.index[i].strftime(fmt_) for i in ticks])
    plt.setp(ax.get_xticklabels(), visible=False); plt.setp(ar.get_xticklabels(), visible=False)
    if fig_only:
        return fig
    fn = f"{sym}_{r['tf']}.png"
    fig.savefig(fn, dpi=130, facecolor=BG, bbox_inches="tight")
    con.print(f"Chart saved: [bold]{fn}[/]")
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------ MARKET-WIDE INTRADAY SCANNER
def load_daily(sym):
    """Daily bars only, quiet. Returns DataFrame or None."""
    try:
        import pypsx_toolkit as pt
        t = pt.PSXTicker(sym)
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=500)
        for fn in (lambda: t.get_historical(start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d")),
                   lambda: t.history(period="2y", interval="1d")):
            try:
                cand = _norm(fn())
            except Exception:
                continue
            if {"open", "close"} <= set(cand.columns):
                return _ohlcv(cand, allow_synth=True)
    except Exception:
        pass
    return None


def list_symbols():
    import pypsx_toolkit as pt
    try:
        x = pt.get_symbols()
    except Exception as e:
        print(f"[!] Could not fetch symbol list ({e}). Using a built-in list of big stocks.")
        return sorted(NAMES)
    if isinstance(x, pd.DataFrame):
        col = next((c for c in x.columns if str(c).lower() in ("symbol", "ticker", "code")), x.columns[0])
        x = x[col].tolist()
    elif isinstance(x, dict):
        x = list(x.keys())
    out = []
    for i in x:
        if isinstance(i, dict):
            i = i.get("symbol") or i.get("ticker") or ""
        out.append(str(i).strip().upper())
    return sorted({s for s in out if re.fullmatch(r"[A-Z0-9]{2,8}", s)})   # drops futures / odd instruments


def load_demo_ticks(sym):
    import zlib
    rng = np.random.default_rng(zlib.crc32(sym.encode()))
    mode = rng.choice(["up", "up", "down", "flat", "flat"])
    base = float(rng.uniform(30, 400))
    rows, price = [], base
    days = pd.bdate_range(end="2026-11-20", periods=3)
    for k, day in enumerate(days):
        drift = (0.00004 if mode == "up" else -0.00004 if mode == "down" else 0.0) if k == len(days) - 1 else 0.0
        for ts in pd.date_range(day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=15, minutes=25), freq="20s"):
            price *= 1 + rng.normal(drift, 0.0006)
            boost = 2.5 if (mode == "up" and k == len(days) - 1 and ts.hour >= 14) else 1
            rows.append((ts, price, int(rng.integers(200, 6000) * boost * (3 if rng.random() < .03 else 1))))
    return pd.DataFrame(rows, columns=["time", "price", "volume"]).set_index("time")


def fetch_ticks(sym, demo):
    if demo:
        return load_demo_ticks(sym)
    import pypsx_toolkit as pt
    for _ in range(2):
        try:
            return _norm(pt.PSXTicker(sym).intraday())
        except Exception:
            time.sleep(0.6)
    return None


def setups_5m(d):
    """Pro-style intraday setups on 5m candles. Returns (list of (name, points, note), info)."""
    out = []
    c, px, atr = d.close, d.close.iloc[-1], d.atr.iloc[-1]
    day = np.asarray(d.index.date)
    today = day[-1]
    td, pv = d[day == today], d[day < today]
    mean_v = td.volume.mean() if len(td) else d.volume.mean()
    rv = float(d.volume.iloc[-3:].mean() / mean_v) if mean_v and mean_v > 0 else float("nan")
    info = dict(rv=rv, orh=None, orl=None, pdh=None, pdl=None, gap=None)
    if len(pv):
        info["pdh"], info["pdl"] = float(pv.high.max()), float(pv.low.min())
        info["gap"] = float(100 * (td.open.iloc[0] / pv.close.iloc[-1] - 1)) if len(td) else None
    if len(td) >= 8:                                           # opening range = first 6 candles (30 min)
        orh, orl = float(td.high.iloc[:6].max()), float(td.low.iloc[:6].min())
        info["orh"], info["orl"] = orh, orl
        recent_in = td.close.iloc[:-1].tail(6).min() <= orh
        recent_in_s = td.close.iloc[:-1].tail(6).max() >= orl
        vol_ok = not np.isnan(rv) and rv >= 1.2
        if px > orh and recent_in and px - orh <= 1.5 * atr:
            out.append(("Opening-range breakout", 20 if vol_ok else 12, f"above {orh:.2f}" + (" on volume" if vol_ok else "")))
        if px < orl and recent_in_s and orl - px <= 1.5 * atr:
            out.append(("Opening-range breakdown", -20 if vol_ok else -12, f"below {orl:.2f}"))
    if info["pdh"] is not None:
        if info["pdh"] < px <= info["pdh"] + 1.5 * atr:
            out.append(("Prev-day high break", 10, f"above {info['pdh']:.2f}"))
        if info["pdl"] - 1.5 * atr <= px < info["pdl"]:
            out.append(("Prev-day low break", -10, f"below {info['pdl']:.2f}"))
    v = d.vwap
    if v.notna().iloc[-1] and len(d) > 6:
        if (c.iloc[-5:-1] < v.iloc[-5:-1]).sum() >= 3 and px > v.iloc[-1]:
            out.append(("VWAP reclaim", 15, f"back above {v.iloc[-1]:.2f}"))
        if (c.iloc[-5:-1] > v.iloc[-5:-1]).sum() >= 3 and px < v.iloc[-1]:
            out.append(("VWAP rejection", -15, f"lost {v.iloc[-1]:.2f}"))
    e20, e50, lo4 = d.ema20.iloc[-1], d.ema50.iloc[-1], d.low.iloc[-4:]
    if e20 > e50 and px > e50 and (lo4 <= d.ema20.iloc[-4:] + 0.3 * atr).any() and c.iloc[-1] > d.open.iloc[-1] and px > e20:
        out.append(("EMA20 pullback in uptrend", 15, f"bounce off {e20:.2f}"))
    hi4 = d.high.iloc[-4:]
    if e20 < e50 and px < e50 and (hi4 >= d.ema20.iloc[-4:] - 0.3 * atr).any() and c.iloc[-1] < d.open.iloc[-1] and px < e20:
        out.append(("EMA20 rejection in downtrend", -15, f"fell from {e20:.2f}"))
    if len(d) > 14:                                            # flag: impulse then tight pause
        leg, cons = c.iloc[-5] - c.iloc[-13], d.high.iloc[-4:].max() - d.low.iloc[-4:].min()
        if leg >= 2.0 * atr and cons <= 1.2 * atr:
            out.append(("Bull flag", 10, f"break above {d.high.iloc[-4:].max():.2f}"))
        if leg <= -2.0 * atr and cons <= 1.2 * atr:
            out.append(("Bear flag", -10, f"break below {d.low.iloc[-4:].min():.2f}"))
    bw = (d.bb_up - d.bb_lo) / d.bb_mid
    if bw.notna().sum() > 40 and bw.iloc[-6:-1].min() <= bw.dropna().iloc[-100:].quantile(.15):
        if px > d.bb_up.iloc[-1]:
            out.append(("Squeeze breakout", 12, "Bollinger band expansion up"))
        if px < d.bb_lo.iloc[-1]:
            out.append(("Squeeze breakdown", -12, "Bollinger band expansion down"))
    return out, info


def evaluate(sym, raw, min_turnover):
    try:
        c5 = candles_from(raw, "5min")
    except Exception:
        return None, "bad data"
    if len(c5) < MIN_BARS["5m"]:
        return None, "too few candles"
    day = np.asarray(c5.index.date)
    has_vol = bool(c5.volume.sum() > 0)
    turn = float((c5.close * c5.volume)[day == day[-1]].sum())
    if has_vol and turn < min_turnover:
        return None, "illiquid"
    r5 = analyse("5m", c5)
    c15 = candles_from(raw, "15min")
    r15 = analyse("15m", c15) if len(c15) >= MIN_BARS["15m"] else None
    c30 = candles_from(raw, "30min")
    r30 = analyse("30m", c30) if len(c30) >= MIN_BARS["30m"] else None
    d = r5["d"]
    x, px = d.iloc[-1], d.close.iloc[-1]
    sl_, info = setups_5m(d)
    pts = sum(p for _, p, _ in sl_)
    for n, b, note in r5["chart"]:
        if "CONFIRMED" in note:
            pts += 10 if b == "bull" else -10 if b == "bear" else 0
        elif n == "Ascending triangle":
            pts += 6
        elif n == "Descending triangle":
            pts -= 6
    b15 = r15["score"] if r15 else 0.0
    b30 = r30["score"] if r30 else 0.0
    wts = [(0.5, r5["score"])] + ([(0.3, b15)] if r15 else []) + ([(0.2, b30)] if r30 else [])
    base = sum(a * b for a, b in wts) / sum(a for a, _ in wts)
    htf = float(np.mean([r["score"] for r in (r15, r30) if r])) if (r15 or r30) else 0.0
    final = float(np.clip(base + pts, -100, 100))
    plan = make_plan(d, len(d) - 1, 1)
    above = [lv for lv, _ in r5["sr"]["levels"] if lv > px * 1.0005]
    above += [v for v in (info["orh"], info["pdh"]) if v and v > px * 1.0005]
    room = (min(above) - px) / plan["risk"] if above and plan["risk"] > 0 else float("inf")
    atr_pct = 100 * x.atr / px
    brk = any(n.split()[0] in ("Opening-range", "Prev-day", "Squeeze") and p > 0 for n, p, _ in sl_)
    checks = [("Score 40+", final >= 40), ("15m/30m not bearish", htf >= 0),
              ("Above VWAP", bool(pd.isna(x.vwap)) or px > x.vwap), ("RSI 45-75", 45 <= x.rsi <= 75),
              ("Trend strength / breakout", x.adx >= 18 or brk),
              ("Volume pickup", (not has_vol) or bool(np.isnan(info["rv"])) or info["rv"] >= 1.0),
              ("Room to run 1R+", room >= 1.0), ("Volatility OK", 0.12 <= atr_pct <= 3.5)]
    mins_t1 = max(1.0, (abs(plan["t1"] - plan["entry"]) / (0.7 * plan["atr"])) ** 2) * 5
    checks.append(("T1 within ~4h", mins_t1 <= 240))
    need = [k for k, ok in checks if not ok]
    kind = "BUY" if not need else "WATCH" if (final >= 25 and len(need) <= 2) else "BEAR" if (final <= -40 and htf <= 0) else None
    ext = (px - x.ema20) / x.atr if x.atr else 0
    grade = "A" if final >= 65 else "B" if final >= 50 else "C"
    nb = backtest(d, r5["sc"], 1, TFS["5m"][1])
    return dict(sym=sym, kind=kind, px=float(px), final=final, b5=r5["score"], b15=b15, b30=b30, setups=sl_, info=info, checks=checks,
                need=need, plan=plan, room=room, rv=info["rv"], atr_pct=atr_pct, turn=turn, mins_t1=mins_t1, grade=grade,
                ext=ext, ema20=float(x.ema20), r5=r5, r15=r15, r30=r30, last_ts=d.index[-1], trig=float(d.high.iloc[-6:].max()),
                bt=nb, daily=None, note=""), None


def apply_daily(ev, daily):
    """Higher-timeframe filter: daily trend + daily resistance."""
    dd = add_ind(daily)
    x, px = dd.iloc[-1], ev["px"]
    ev["daily"] = "Up" if x.close > x.ema20 > x.ema50 else "Down" if x.close < x.ema20 < x.ema50 else "Mixed"
    res = [lv for lv, _ in sr_levels(dd)["levels"] if lv > px * 1.0005]
    if ev["kind"] == "BUY":
        if res and (min(res) - px) < ev["plan"]["risk"]:
            ev["kind"], ev["need"] = "WATCH", ev["need"] + ["Daily resistance %.2f too close" % min(res)]
        elif ev["daily"] == "Down":
            ev["grade"] = {"A": "B", "B": "C", "C": "C"}[ev["grade"]]
            ev["note"] = "counter daily trend"
        elif ev["daily"] == "Up":
            ev["note"] = "with daily uptrend"


def fmt_hold(m):
    return f"~{m:.0f} min" if m < 90 else f"~{m / 60:.1f} h"


def scan_market(symbols, demo, top, min_turnover, workers, want_news, limit, progress=None):
    """Compute only (no printing). Returns a dict of results."""
    t0 = time.time()
    uni = symbols or (sorted({f"DEM{i:02d}" for i in range(60)}) if demo else list_symbols())
    if limit:
        uni = uni[:limit]
    if progress:
        progress(0, len(uni))
    raws, done = {}, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, raw in zip(uni, ex.map(lambda s: fetch_ticks(s, demo), uni)):
            done += 1
            if raw is not None and len(raw):
                raws[sym] = raw
            if progress:
                progress(done, len(uni))
    skipped, evs = {}, []
    for sym, raw in raws.items():
        try:
            ev, why = evaluate(sym, raw, min_turnover)
        except Exception as e:
            ev, why = None, f"error: {str(e)[:30]}"
        if ev:
            evs.append(ev)
        else:
            skipped[why] = skipped.get(why, 0) + 1
    skipped["no intraday data"] = len(uni) - len(raws)
    if not evs:
        return dict(evs=[], skipped=skipped, uni=uni, t0=t0)
    # pooled history of the 5m long setup across everything scanned (small window, honest label)
    W = L = T = 0
    for ev in evs:
        b = ev["bt"]
        if b["n"]:
            W += round(b["win"] * b["n"] / 100); L += round(b["loss"] * b["n"] / 100); T += b["n"] - round(b["win"] * b["n"] / 100) - round(b["loss"] * b["n"] / 100)
    N = W + L + T
    # stage 2: daily context for the best candidates
    cand = sorted([e for e in evs if e["kind"] in ("BUY", "WATCH")], key=lambda e: -e["final"])[:max(top * 3, 20)]
    for ev in cand:
        dly = load_demo(ev["sym"])[0] if demo else load_daily(ev["sym"])
        if dly is not None and len(dly) > 55:
            apply_daily(ev, dly)
    buys = sorted([e for e in evs if e["kind"] == "BUY"], key=lambda e: (-"CBA".index(e["grade"]), e["final"]), reverse=True)[:top]
    buys.sort(key=lambda e: ("ABC".index(e["grade"]), -e["final"]))
    watch = sorted([e for e in evs if e["kind"] == "WATCH"], key=lambda e: -e["final"])[:8]
    bears = sorted([e for e in evs if e["kind"] == "BEAR"], key=lambda e: e["final"])[:6]
    nw = {}
    if want_news and not demo:
        for e in buys[:5]:
            nw[e["sym"]] = get_news(e["sym"])
    last_ts, age = max(e["last_ts"] for e in evs), None
    if not demo:
        try:
            now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo else pd.Timestamp.now()
            age = (now - last_ts).total_seconds() / 60
        except Exception:
            age = None
    return dict(uni=uni, evs=evs, skipped=skipped, pool=(W, L, T, N), buys=buys, watch=watch, bears=bears, nw=nw,
                last_ts=last_ts, age=age, t0=t0)


def market_scan(symbols, demo, top, min_turnover, workers, capital, risk_pct, want_news, show, limit):
    def prog(done, total):
        if done == 0:
            con.print(f"[bold cyan]Scanning {total} symbols...[/] (this can take a few minutes)")
        elif done % 50 == 0:
            con.print(f"  [dim]downloaded {done}/{total}[/]")
    R = scan_market(symbols, demo, top, min_turnover, workers, want_news, limit, prog)
    if not R["evs"]:
        con.print(Panel("No stock had usable intraday data. Likely market closed with no ticks, or the feed returned nothing.\n"
                        f"Skipped: {R['skipped']}", title="NO DATA", border_style="red"))
        return
    uni, evs, skipped, (W, L, T, N) = R["uni"], R["evs"], R["skipped"], R["pool"]
    buys, watch, bears, nw, last_ts, t0 = R["buys"], R["watch"], R["bears"], R["nw"], R["last_ts"], R["t0"]
    stale = ""
    if R["age"] and R["age"] > 45:
        stale = (f"\n[bold yellow]Latest tick is {R['age'] / 60:.1f} h old ({last_ts:%d %b %H:%M}). Market looks closed: "
                 f"these are setups to confirm at the NEXT open, not live entries.[/]")
    pool = ""
    if N >= 30:
        exp = (1.5 * W - L) / N
        col = "green" if exp > 0.05 else "red" if exp < -0.05 else "yellow"
        pool = (f"\nSame 5m long rule across all scanned stocks (data window only): T1 hit {100 * W / N:.0f}%, SL hit {100 * L / N:.0f}%, "
                f"[{col}]avg {exp:+.2f}R[/] over {N} past signals" + ("  [yellow](weak edge in this window: size down)[/]" if exp <= 0 else ""))
    con.print()
    con.print(Panel(f"[bold]INTRADAY SIGNAL SCAN[/]   data to {last_ts:%d %b %Y %H:%M}\n"
                    f"Universe {len(uni)} | analysed {len(evs)} | skipped {sum(skipped.values())} "
                    f"({', '.join(f'{k} {v}' for k, v in skipped.items() if v)})" + pool + stale,
                    border_style="cyan", box=box.ROUNDED))
    # ---- signals
    if buys:
        t = Table(box=box.SIMPLE_HEAD, title="🟢 INTRADAY BUY SIGNALS (best first)", title_style="bold green", expand=True)
        for c in ("#", "Gr", "Stock", "Price", "Score", "Setup", "Entry", "SL", "T1", "T2", "Hold to T1", "RVol", "Daily"):
            t.add_column(c)
        for i, e in enumerate(buys, 1):
            p = e["plan"]
            t.add_row(str(i), f"[bold]{e['grade']}[/]", f"[bold]{esc(e['sym'])}[/]", f"{e['px']:,.2f}", f"{e['final']:+.0f}",
                      esc(", ".join(n for n, pt_, _ in e["setups"] if pt_ > 0)[:34] or "trend + momentum"),
                      f"{p['entry']:,.2f}", f"[red]{p['sl']:,.2f}[/]", f"[green]{p['t1']:,.2f}[/]", f"[green]{p['t2']:,.2f}[/]",
                      fmt_hold(e["mins_t1"]), "-" if np.isnan(e["rv"]) else f"{e['rv']:.1f}x", e["daily"] or "-")
        con.print(t)
        for e in buys[:3]:
            p, r5 = e["plan"], e["r5"]
            lines = ["  ".join(f"{'[green]✔[/]' if ok else '[red]✘[/]'} {k}" for k, ok in e["checks"])]
            lines.append("Setups: " + esc("; ".join(f"{n} ({nt})" for n, pt_, nt in e["setups"] if pt_ > 0) or "none"))
            cp = [f"{esc(n)} ({b})" for n, b, _ in r5["chart"]]
            lines.append("Candles: " + esc(", ".join(r5["candles"]) or "none") + "   Patterns: " + (", ".join(cp) or "none"))
            fc = r5["fc"]
            lines.append(f"Plan: entry {p['entry']:,.2f}, SL {p['sl']:,.2f} (risk {100 * p['risk'] / p['entry']:.2f}%), "
                         f"T1 {p['t1']:,.2f} in {fmt_hold(e['mins_t1'])}, T2 {p['t2']:,.2f} in {fmt_hold(e['mins_t1'] * 2.8)}. "
                         f"[yellow]Time-stop: if T1 is not hit within ~{fmt_hold(2 * e['mins_t1'])}, exit. Square off before the market closes.[/]")
            lines.append(f"Next ~1 hour outlook: {fc['bias']} ({fc['pu']:.0f}% up), range {fc['lo']:,.2f} to {fc['hi']:,.2f}")
            if e["ext"] > 1.8:
                lines.append(f"[yellow]Extended above EMA20: safer entry is a pullback near {e['ema20']:,.2f}.[/]")
            if e["note"]:
                lines.append(f"Daily context: {e['note']}")
            if capital:
                lines.append(f"Size: risk {risk_pct}% of {capital:,.0f} = {capital * risk_pct / 100:,.0f} PKR -> about {int(capital * risk_pct / 100 / p['risk'])} shares")
            n_ = nw.get(e["sym"])
            if n_ and not n_["err"] and n_["stock"]:
                lab, col = news_label(n_["score"])
                lines.append(f"News: [bold {col}]{lab}[/] ({n_['score']:+.0f}): " + esc(n_["stock"][0]["title"][:80]))
                if n_["score"] <= -25:
                    lines.append("[yellow]News is negative, conflicts with this BUY. Skip or cut size.[/]")
            con.print(Panel("\n".join(lines), title=f"{e['grade']}  {esc(e['sym'])}  {e['px']:,.2f}", border_style="green", box=box.ROUNDED))
    else:
        con.print(Panel("[bold]No stock passes all 9 checks right now.[/] That is a normal outcome: most of the day offers no clean setup.\n"
                        "Watch the list below and wait for triggers.", title="🟢 INTRADAY BUY SIGNALS", border_style="yellow"))
    if watch:
        w = Table(box=box.SIMPLE_HEAD, title="👀 WATCHLIST (almost there)", title_style="bold yellow", expand=True)
        for c in ("Stock", "Price", "Score", "Trigger", "Still needs"):
            w.add_column(c)
        for e in watch:
            w.add_row(f"[bold]{esc(e['sym'])}[/]", f"{e['px']:,.2f}", f"{e['final']:+.0f}", f"close above {e['trig']:,.2f}", esc(", ".join(e["need"])))
        con.print(w)
    if bears:
        con.print("[bold red]🔴 Weak / avoid or exit:[/] " + ", ".join(f"{esc(e['sym'])} ({e['final']:+.0f})" for e in bears)
                  + "  [dim](retail shorting is usually impractical on PSX)[/]")
    # ---- save + charts
    rows = [dict(symbol=e["sym"], type=e["kind"], grade=e["grade"], price=e["px"], score=round(e["final"]), entry=e["plan"]["entry"],
                 sl=e["plan"]["sl"], t1=e["plan"]["t1"], t2=e["plan"]["t2"], hold_to_t1_min=round(e["mins_t1"]),
                 setups="; ".join(n for n, p_, _ in e["setups"] if p_ > 0), need="; ".join(e["need"]), daily=e["daily"])
            for e in buys + watch]
    if rows:
        fn = f"psx_signals_{pd.Timestamp.now():%Y%m%d_%H%M}.csv"
        pd.DataFrame(rows).to_csv(fn, index=False)
        con.print(f"Saved: [bold]{fn}[/]")
    for e in buys[:3]:
        r = dict(e["r5"]); r.update(plan=e["plan"], side=1, score=e["final"])
        draw(e["sym"], r, show)
    con.print(f"[dim]Done in {time.time() - t0:.0f}s. Educational use only, not financial advice. Always use the stop loss. "
              f"Re-run every 15-30 minutes during market hours.[/]\n")


# ------------------------------------------------------------------ main
def analyse_stock(sym, demo=False, name=None, news=True):
    """Compute only (no printing). Used by the console script AND the web app."""
    sym = sym.upper()
    daily, raw = load_demo(sym) if demo else load_live(sym)
    frames, why, notes, tick_info = {"1D": daily}, {}, [], ""
    has_raw = raw is not None and len(raw) > 0
    if not has_raw:
        for tf in ("5m", "15m", "30m", "1h"):
            why[tf] = "no intraday data loaded" + (f" ({INTRADAY_ERR})" if INTRADAY_ERR else " (empty response)")
    else:
        tick_info = f"Intraday ticks loaded: {len(raw):,} rows, {raw.index.min():%d %b %H:%M} to {raw.index.max():%d %b %H:%M}"
        for tf, rule in (("5m", "5min"), ("15m", "15min"), ("30m", "30min"), ("1h", "60min")):
            try:
                frames[tf] = candles_from(raw, rule)
            except Exception as e:
                why[tf] = f"could not build candles: {str(e)[:100]}"
    if not demo:
        h = load_hourly_history(sym)
        if h is not None and len(h) > len(frames.get("1h", [])):
            frames["1h"] = h
            why.pop("1h", None)
            notes.append(f"1h history loaded via pypsx key ({len(h)} candles)")
    psess = prev_sessions(daily, raw.index[-1].date() if has_raw else None)
    extra = [v for p in psess for v in (p["high"], p["low"])]
    results = []
    for tf in TFS:
        if tf not in frames:
            continue
        n = len(frames[tf])
        if n < MIN_BARS[tf]:
            mins = TFS[tf][2]
            wait = f" about {(MIN_BARS[tf] - n) * mins} more minutes of trading needed." if mins else ""
            why[tf] = f"only {n} candles so far (need {MIN_BARS[tf]}).{wait}"
            continue
        results.append(analyse(tf, frames[tf], extra if TFS[tf][0] == "Intraday" or tf == "1h" else None))
    if not results:
        raise ValueError("No usable data for " + sym)
    nw = (demo_news() if demo else get_news(sym, name)) if news else None
    return dict(sym=sym, results=results, why=why, notes=notes, tick_info=tick_info, nw=nw, psess=psess)


def scan(sym, demo=False, capital=0, risk_pct=1.0, show=True, name=None, news=True):
    try:
        out = analyse_stock(sym, demo, name, news)
    except ValueError as e:
        sys.exit(str(e))
    if out["tick_info"]:
        con.print(f"[dim]{out['tick_info']}[/]")
    for n_ in out["notes"]:
        con.print(f"[dim]{esc(n_)}[/]")
    results = out["results"]
    best = report(out["sym"], results, out["nw"], capital, risk_pct, out["why"], out["psess"])
    draw(out["sym"], best or max(results, key=lambda r: abs(r["score"])), show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--scan", action="store_true", help="MARKET SCAN: find intraday signals across many stocks")
    ap.add_argument("--top", type=int, default=10, help="max BUY signals to list (scan mode)")
    ap.add_argument("--min-turnover", type=float, default=20_000_000, help="min last-session turnover in PKR (scan mode)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="scan only the first N symbols (testing)")
    ap.add_argument("--news", action="store_true", help="also check news for the top signals (scan mode)")
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    ap.add_argument("--capital", type=float, default=0)
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--name", default=None, help="company name for better news matching")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    a, _ = ap.parse_known_args()   # ignore extra args like Colab's -f
    syms = [x.upper() for x in a.symbols if not x.startswith("-") and not x.endswith(".json")]
    if a.scan:
        market_scan(syms, a.demo, a.top, a.min_turnover, a.workers, a.capital, a.risk_pct, a.news, not a.no_show, a.limit)
    else:
        syms = syms or input("Stock symbol(s), e.g. OGDC PPL: ").upper().split()
        for s_ in syms:
            scan(s_, a.demo, a.capital, a.risk_pct, not a.no_show, a.name, not a.no_news)
