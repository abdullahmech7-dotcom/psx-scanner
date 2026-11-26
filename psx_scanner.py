"""PSX-first market scanner. Uses pypsx-toolkit first and Yahoo Finance fallback.
Signals are heuristic, not investment advice. Demo data is never used implicitly.
"""
from __future__ import annotations
import re, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

PSX_UNIVERSE = sorted(set('''AABS ABOT AICL AIRLINK AKBL APL ARL ATRL AVN BAFL BAHL BOP BOK CNERGY DGKC DOL DSFL EFERT ENGRO EXIDE FCCL FFC FFBL GHNI GHNL GATM HBL HUBC IBFL INDU ISL KAPCO KEL KOHC LUCK MARI MEBL MLCF MTL NBP NRL OGDC PAKRI PAEL PPL POL PRL PSO SNGP SSGC SYS TPLP TRG UBL UNITY WTL'''.split()))


def normalize(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty: return None
    d=df.copy(); d.columns=[str(c).strip().lower().replace(' ','_') for c in d.columns]
    aliases={'adj_close':'close','vol':'volume','datetime':'date','timestamp':'date'}
    d.rename(columns={c:aliases[c] for c in d.columns if c in aliases},inplace=True)
    if not isinstance(d.index,pd.DatetimeIndex):
        col=next((c for c in ('date','time') if c in d.columns),None)
        if col: d.index=pd.to_datetime(d.pop(col),errors='coerce')
        else: d.index=pd.to_datetime(d.index,errors='coerce')
    d=d[~d.index.isna()].sort_index(); d=d[~d.index.duplicated(keep='last')]
    for c in ['open','high','low','close']:
        if c not in d: return None
    if 'volume' not in d: d['volume']=0
    return d[['open','high','low','close','volume']].apply(pd.to_numeric,errors='coerce').dropna(subset=['close'])


def fetch_daily(symbol, period='1y'):
    # PSX toolkit preferred. API shape varies across releases, so guarded attempts.
    try:
        import pypsx_toolkit as pt
        t=pt.PSXTicker(symbol)
        for f in [lambda:t.history(period=period,interval='1d'), lambda:t.get_historical(start_date=(pd.Timestamp.today()-pd.Timedelta(days=400)).strftime('%Y-%m-%d'),end_date=pd.Timestamp.today().strftime('%Y-%m-%d'))]:
            try:
                d=normalize(f())
                if d is not None and len(d)>30: return d
            except Exception: pass
    except Exception: pass
    # Yahoo Finance fallback for PSX symbols
    try:
        import yfinance as yf
        for ticker in [symbol+'.KA',symbol]:
            d=normalize(yf.download(ticker,period=period,interval='1d',auto_adjust=False,progress=False,threads=False))
            if d is not None and len(d)>30:return d
    except Exception: pass
    return None


def fetch_intraday(symbol, period='5d', interval='5m'):
    try:
        import pypsx_toolkit as pt
        raw=pt.PSXTicker(symbol).intraday()
        d=normalize(raw)
        if d is not None:return d
        if isinstance(raw,pd.DataFrame) and not raw.empty:
            r=raw.copy();r.columns=[str(c).lower() for c in r.columns]
            price=next((c for c in ['price','last','close','rate'] if c in r),None)
            if price:
                if not isinstance(r.index,pd.DatetimeIndex): r.index=pd.to_datetime(r.get('time',r.index),errors='coerce')
                o=r[price].resample(interval).ohlc(); vol=r.get('volume',pd.Series(0,index=r.index)).resample(interval).sum();o['volume']=vol;return normalize(o)
    except Exception: pass
    try:
        import yfinance as yf
        d=normalize(yf.download(symbol+'.KA',period=period,interval=interval,auto_adjust=False,progress=False,threads=False))
        if d is not None:return d
    except Exception:pass
    return None


def add_indicators(d):
    x=d.copy(); c=x.close
    x['ema9']=c.ewm(span=9,adjust=False).mean();x['ema20']=c.ewm(span=20,adjust=False).mean();x['ema50']=c.ewm(span=50,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0);loss=-delta.clip(upper=0);rs=gain.ewm(alpha=1/14,adjust=False).mean()/loss.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan);x['rsi']=100-100/(1+rs)
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean();x['macd']=m;x['macd_sig']=m.ewm(span=9,adjust=False).mean();x['hist']=m-x['macd_sig']
    tr=pd.concat([x.high-x.low,(x.high-c.shift()).abs(),(x.low-c.shift()).abs()],axis=1).max(axis=1);x['atr']=tr.ewm(alpha=1/14,adjust=False).mean()
    x['vol_ma']=x.volume.rolling(20,min_periods=5).mean();x['vol_ratio']=x.volume/x.vol_ma.replace(0,np.nan)
    x['bb_mid']=c.rolling(20).mean();std=c.rolling(20).std();x['bb_up']=x.bb_mid+2*std;x['bb_lo']=x.bb_mid-2*std
    x['hh20']=x.high.shift(1).rolling(20).max();x['ll20']=x.low.shift(1).rolling(20).min()
    return x


def candle_patterns(d):
    o,h,l,c=d.open,d.high,d.low,d.close; body=(c-o).abs();rng=(h-l).replace(0,np.nan);uw=h-np.maximum(o,c);lw=np.minimum(o,c)-l
    pats=[]
    def latest(cond,name,bias):
        try:
            if bool(cond.iloc[-1]):pats.append((name,bias))
        except Exception:pass
    latest((lw>=2*body)&(uw<=.35*body)&(c>o),'Hammer','bull')
    latest((uw>=2*body)&(lw<=.35*body)&(c<o),'Shooting star','bear')
    latest((c>o)&(o<=c.shift(1))&(c>=o.shift(1))&(c.shift(1)<o.shift(1)),'Bullish engulfing','bull')
    latest((c<o)&(o>=c.shift(1))&(c<=o.shift(1))&(c.shift(1)>o.shift(1)),'Bearish engulfing','bear')
    latest((body/rng<.1),'Doji','neutral')
    latest((c>o)&(body>body.rolling(10).mean())&(body>=.8*rng),'Bullish marubozu','bull')
    latest((c<o)&(body>body.rolling(10).mean())&(body>=.8*rng),'Bearish marubozu','bear')
    return pats


def chart_patterns(d):
    x=d.tail(60); out=[]
    # Approximate pivot-based double tops/bottoms, triangles, channel breakouts
    highs=x.high; lows=x.low; a=x.atr.iloc[-1];
    if len(x)>20 and pd.notna(a) and a>0:
        p_hi=highs[(highs.shift(2)<highs)&(highs.shift(1)<highs)&(highs.shift(-1)<highs)&(highs.shift(-2)<highs)]
        p_lo=lows[(lows.shift(2)>lows)&(lows.shift(1)>lows)&(lows.shift(-1)>lows)&(lows.shift(-2)>lows)]
        if len(p_hi)>=2 and abs(p_hi.iloc[-1]-p_hi.iloc[-2])<a:out.append(('Potential double top','bear'))
        if len(p_lo)>=2 and abs(p_lo.iloc[-1]-p_lo.iloc[-2])<a:out.append(('Potential double bottom','bull'))
        if x.close.iloc[-1]>x.hh20.iloc[-1]:out.append(('20-bar breakout','bull'))
        if x.close.iloc[-1]<x.ll20.iloc[-1]:out.append(('20-bar breakdown','bear'))
        if len(p_hi)>=2 and len(p_lo)>=2:
            if p_hi.iloc[-1]<=p_hi.iloc[-2] and p_lo.iloc[-1]>p_lo.iloc[-2]:out.append(('Ascending triangle','bull'))
            elif p_hi.iloc[-1]<p_hi.iloc[-2] and p_lo.iloc[-1]>=p_lo.iloc[-2]:out.append(('Descending triangle','bear'))
    return out


def news(symbol, company=None):
    terms=[symbol+' PSX']
    if company:terms.insert(0,company)
    items=[]
    for q in terms:
        url='https://news.google.com/rss/search?q='+urllib.parse.quote(q+' when:7d')+'&hl=en-PK&gl=PK&ceid=PK:en'
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'});root=ET.fromstring(urllib.request.urlopen(req,timeout=6).read())
            for it in root.findall('.//item')[:10]:
                title=(it.findtext('title') or '').strip();source=(it.findtext('source') or 'News').strip();date=it.findtext('pubDate') or ''
                if title and title not in [i['title'] for i in items]:items.append({'title':title,'source':source,'date':date})
        except Exception:continue
    return items[:8]


def news_risk(items):
    # keyword heuristic; deliberately conservative and transparent
    neg=['investigation','fraud','default','suspend','shutdown','loss','fire','strike','lawsuit','penalty','probe','downgrade','delay','crisis','attack','sanction','shortage','leak','rumour','rumor']
    pos=['profit','record','contract','award','acquisition','dividend','upgrade','expansion','growth','approval','agreement','launch']
    score=0; details=[]
    for it in items:
        t=it['title'].lower();n=sum(k in t for k in neg);p=sum(k in t for k in pos)
        if n:score-=min(2,n);details.append(('negative',it['title']))
        elif p:score+=min(1,p);details.append(('positive',it['title']))
    return int(np.clip(score,-5,5)),details


def analyse(symbol, horizon='intraday', company=None, include_news=True):
    daily=fetch_daily(symbol)
    if daily is None or len(daily)<40:return None
    d=add_indicators(daily);x=d.iloc[-1];px=float(x.close);atr=float(x.atr)
    score=0;checks=[]
    score+=20 if px>float(x['ema20']) else -15;score+=15 if float(x['ema20'])>float(x['ema50']) else -15;score+=10 if float(x['hist'])>0 else -10
    score+=int(np.clip((float(x['rsi'])-50)*.45,-20,20));score+=10 if px>float(x['hh20']) else (-10 if px<float(x['ll20']) else 0);score+=5 if pd.notna(x['vol_ratio']) and float(x['vol_ratio'])>1.5 else 0
    cp=candle_patterns(d);pat=chart_patterns(d);score+=sum(8 if b=='bull' else -8 if b=='bear' else 0 for _,b in cp);score+=sum(6 if b=='bull' else -6 if b=='bear' else 0 for _,b in pat)
    if horizon=='intraday':
        intra=fetch_intraday(symbol)
        if intra is not None and len(intra)>10:
            ii=add_indicators(intra);z=ii.iloc[-1];score+=15 if float(z['close'])>float(z['ema20']) else -10;score+=10 if float(z['hist'])>0 else -10;score+=8 if pd.notna(z['vol_ratio']) and float(z['vol_ratio'])>1.4 else 0
            # session VWAP estimate
            tp=(ii.high+ii.low+ii.close)/3; day=np.asarray(ii.index.date);last=day[-1];td=ii[day==last];v=td.volume
            vwap=float((tp[day==last]*v).sum()/v.sum()) if v.sum()>0 else np.nan
            if pd.notna(vwap):score+=8 if z.close>vwap else -8
        else:intra=None;vwap=np.nan
    else: intra=None;vwap=np.nan
    nitems=news(symbol,company) if include_news else [];nr,nd=news_risk(nitems);score+=nr*5
    score=float(np.clip(score,-100,100)); side='BUY' if score>=35 else 'WATCH' if score>=15 else 'BEAR' if score<=-35 else 'NO TRADE'
    entry=px;stop=px-1.5*atr;target1=px+2*atr;target2=px+3*atr
    if side=='BEAR':stop=px+1.5*atr;target1=px-2*atr;target2=px-3*atr
    risk=abs(entry-stop);rr=abs(target1-entry)/risk if risk else 0
    resistance=float(d.high.tail(30).max());support=float(d.low.tail(30).min())
    if side=='BUY' and resistance>px and (resistance-px)<risk:side='WATCH';checks.append('Nearby resistance may limit upside')
    if side=='BUY' and nr<=-2:side='WATCH';checks.append('Negative news-risk filter downgraded setup')
    return {'symbol':symbol,'price':px,'score':score,'signal':side,'horizon':horizon,'entry':entry,'stop':stop,'target1':target1,'target2':target2,'risk_reward':rr,'atr':atr,'rsi':float(x['rsi']),'trend':'Bullish' if px>x['ema20']>x['ema50'] else 'Bearish' if px<x['ema20']<x['ema50'] else 'Mixed','volume_ratio':float(x['vol_ratio']) if pd.notna(x['vol_ratio']) else 0,'candles':cp,'patterns':pat,'news':nitems,'news_risk':nr,'warnings':checks,'support':support,'resistance':resistance,'data_date':str(d.index[-1]),'daily':d,'intraday':intra}


def scan_market(symbols=None, horizon='intraday', workers=5, include_news=True):
    universe=list(symbols or PSX_UNIVERSE)
    results=[]
    # Prevent a single malformed ticker or provider response from aborting the full scan.
    def safe_analyse(symbol):
        try:
            return analyse(symbol, horizon, include_news=include_news)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=max(1,int(workers))) as ex:
        results=list(ex.map(safe_analyse, universe))
    results=[r for r in results if r is not None]
    results.sort(key=lambda r:(r['signal']=='BUY', r['score'], r['risk_reward']), reverse=True)
    return results
