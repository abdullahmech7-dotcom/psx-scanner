import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import psx_scanner as ps

st.set_page_config(page_title='PSX Smart Scanner',page_icon='📈',layout='wide')
st.title('📈 PSX Smart Scanner')
st.caption('Pakistan Stock Exchange focused • technical setups, patterns, news risk and trade levels • educational tool')

with st.sidebar:
    st.header('Scan settings')
    market=st.selectbox('Market',['Pakistan Stock Exchange (PSX)','Crypto','Forex'],index=0)
    horizon=st.selectbox('Holding horizon',['Intraday','1–2 days','3–4 days','5 days','1 week','2 weeks','1 month'],index=0)
    hmap={'Intraday':'intraday','1–2 days':'swing','3–4 days':'swing','5 days':'swing','1 week':'swing','2 weeks':'swing','1 month':'swing'}
    horizon_key=hmap[horizon]
    include_news=st.checkbox('News/event risk filter',True)
    scan_limit=st.slider('PSX scan size (universe order)',10,len(ps.PSX_UNIVERSE),len(ps.PSX_UNIVERSE),step=10)
    workers=st.slider('Parallel data workers',1,10,5)
    st.caption('Capital is not required. Levels are calculated independently of account size.')

if market!='Pakistan Stock Exchange (PSX)':
    st.warning('PSX is the implemented and prioritized data engine in this version. Crypto/Forex selection is shown but these markets are not yet connected; no simulated results are displayed.')
    st.stop()


def show_result(r):
    cols=st.columns(4)
    cols[0].metric('Signal',r['signal']);cols[1].metric('Score',f"{r['score']:.0f}/100");cols[2].metric('Price',f"Rs {r['price']:.2f}");cols[3].metric('Trend',r['trend'])
    st.write(f"**Holding horizon:** {r['horizon']} • **Data date:** {r['data_date']} • **RSI:** {r['rsi']:.1f} • **Volume ratio:** {r['volume_ratio']:.2f}x")
    cols=st.columns(4);cols[0].metric('Entry reference',f"Rs {r['entry']:.2f}");cols[1].metric('Stop-loss',f"Rs {r['stop']:.2f}");cols[2].metric('Target 1',f"Rs {r['target1']:.2f}");cols[3].metric('Target 2',f"Rs {r['target2']:.2f}")
    st.write(f"**Approximate reward/risk:** {r['risk_reward']:.2f} • **Support:** Rs {r['support']:.2f} • **Resistance:** Rs {r['resistance']:.2f}")
    if r['warnings']:
        for w in r['warnings']:st.warning(w)
    st.markdown('#### Candlestick signals')
    st.write(', '.join(f"{name} ({bias})" for name,bias in r['candles']) or 'No recent candle pattern detected.')
    st.markdown('#### Chart patterns')
    st.write(', '.join(f"{name} ({bias})" for name,bias in r['patterns']) or 'No chart pattern detected.')
    st.markdown('#### News / event risk')
    st.write(f"News risk score: {r['news_risk']} (negative values indicate potential adverse headlines)")
    if r['news']:
        for n in r['news']:st.markdown(f"- **{n['title']}** — {n['source']} ({n['date']})")
    else:st.caption('No matching headlines returned or news feed unavailable.')
    d=r['daily'].tail(100)
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(d.index,d.close,label='Close');ax.plot(d.index,d.ema20,label='EMA20');ax.plot(d.index,d.ema50,label='EMA50');ax.legend();ax.grid(alpha=.25);ax.set_title(r['symbol']);st.pyplot(fig);plt.close(fig)
    st.caption('Technical scores and pattern labels are heuristics, not verified probabilities or guaranteed forecasts. Headlines and rumors may be incomplete or misclassified. Confirm PSX liquidity, disclosures, and live prices before acting.')

mode=st.radio('Mode',['Market scan','Single stock'],horizontal=True)

@st.cache_data(ttl=180,show_spinner=False)
def run_scan(universe,horizon,workers,news):
    return ps.scan_market(universe,horizon,workers,news)

if mode=='Market scan':
    st.info(f'Scanning up to {scan_limit} PSX tickers. The results depend on available data sources; unavailable symbols are omitted.')
    if st.button('🚀 Scan PSX market',type='primary'):
        with st.spinner('Downloading market data and calculating technical setups...'):
            results=run_scan(tuple(ps.PSX_UNIVERSE[:scan_limit]),horizon_key,workers,include_news)
        if not results: st.error('No usable data returned. Try again later or use Single stock mode.');st.stop()
        st.session_state['scan_results']=results
    results=st.session_state.get('scan_results')
    if results:
        df=pd.DataFrame([{k:r[k] for k in ['symbol','signal','score','price','trend','rsi','volume_ratio','risk_reward','news_risk']} for r in results])
        df=df.sort_values(['signal','score'],ascending=[True,False])
        st.subheader('Ranked scan results')
        st.dataframe(df.rename(columns={'symbol':'Ticker','signal':'Signal','score':'Score','price':'Price','trend':'Trend','rsi':'RSI','volume_ratio':'Vol x','risk_reward':'R:R','news_risk':'News risk'}),use_container_width=True,hide_index=True)
        st.download_button('Download results CSV',df.to_csv(index=False).encode(),file_name='psx_scan.csv',mime='text/csv')
        buys=[r for r in results if r['signal']=='BUY']
        if buys:
            st.markdown('### Highest-ranked BUY candidates')
            for r in buys[:10]:
                with st.expander(f"{r['symbol']} • Score {r['score']:.0f} • Rs {r['price']:.2f}",expanded=(r==buys[0])):
                    show_result(r)
        else:st.warning('No BUY setups met the scanner threshold. WATCH/NO TRADE is preferable to forcing a trade.')
else:
    sym=st.text_input('PSX ticker',value='OGDC').strip().upper()
    if st.button('Analyze ticker',type='primary'):
        with st.spinner(f'Analyzing {sym}...'):
            r=ps.analyse(sym,horizon_key,include_news=include_news)
        if not r:st.error('Could not retrieve usable PSX data for this ticker.');st.stop()
        st.session_state['single']=r
    r=st.session_state.get('single')
    if r:show_result(r)

