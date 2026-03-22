import streamlit as st
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline
import datetime
from GoogleNews import GoogleNews

# --- 1. 초기 설정 및 모델 로드 ---
st.set_page_config(page_title="Global AI Stock Predictor", layout="wide")

@st.cache_resource
def load_ai_models():
    # 금융 전문 감성 분석 모델 (영어 기반)
    return pipeline("text-classification", model="ProsusAI/finbert")

sentiment_analyzer = load_ai_models()

@st.cache_data
def get_global_stock_list():
    # 한국 KRX + 미국 S&P500 종목 합치기
    with st.spinner('종목 리스트 업데이트 중...'):
        krx = fdr.StockListing('KRX')[['Code', 'Name']]
        krx['Display'] = krx['Name'] + " (" + krx['Code'] + ".KS)"
        krx['Ticker'] = krx['Code'] + ".KS"
        
        us = fdr.StockListing('S&P500')[['Symbol', 'Name']]
        us['Display'] = us['Name'] + " (" + us['Symbol'] + ")"
        us['Ticker'] = us['Symbol']
        
        full_df = pd.concat([krx[['Display', 'Ticker']], us[['Display', 'Ticker']]])
        return dict(zip(full_df['Display'], full_df['Ticker']))

STOCK_DICT = get_global_stock_list()

# --- 2. 사이드바 UI ---
st.sidebar.header("🔍 분석 설정")
selected_name = st.sidebar.selectbox("종목 검색 (자동완성)", options=list(STOCK_DICT.keys()), index=0)
ticker = STOCK_DICT[selected_name]
period = st.sidebar.select_slider("학습 데이터 기간", options=["3mo", "6mo", "1y", "2y"], value="1y")

# --- 3. 데이터 수집 함수 ---
@st.cache_data(ttl=3600)
def fetch_data(symbol, p):
    df = yf.download(symbol, period=p, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index()

df = fetch_data(ticker, period)

# --- 4. 메인 화면 구성 ---
st.title(f"📈 {selected_name} AI 분석 리포트")

if not df.empty:
    # 차트 레이아웃
    tab1, tab2 = st.tabs(["주가 차트", "AI 예측 상세"])
    
    with tab1:
        fig = px.line(df, x='Date', y='Close', title=f"{ticker} 종가 추이")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🤖 실시간 모델 재학습 및 예측")
        
        # 데이터 가공 (학습용)
        df_ml = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        X = df_ml.iloc[:-1] # 오늘 데이터까지 (문제)
        y = df_ml['Close'].shift(-1).iloc[:-1] # 내일 종가 (정답)
        
        # 모델 학습
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 예측
        last_day_data = df_ml.iloc[[-1]]
        pred_price = model.predict(last_day_data)[0]
        curr_price = df_ml['Close'].iloc[-1]
        diff = pred_price - curr_price
        
        # 결과 표시
        c1, c2, c3 = st.columns(3)
        c1.metric("현재가", f"{curr_price:,.2f}")
        c2.metric("AI 내일 예상가", f"{pred_price:,.2f}", delta=f"{diff:.2f}")
        c3.write("✅ 본 예측은 과거 패턴 기반이며 투자 권유가 아닙니다.")

    # --- 5. 뉴스 분석 섹션 (데모 로직) ---
    st.divider()
st.subheader("📰 실시간 뉴스 감성 분석")

if st.button("실시간 뉴스 분석 실행"):
    with st.spinner(f'{selected_name}의 최신 뉴스를 긁어오는 중...'):
        # 1. 구글 뉴스에서 해당 종목 뉴스 검색 (최근 1일치)
        googlenews = GoogleNews(lang='en', region='US', period='1d') 
        googlenews.search(selected_name)
        news_results = googlenews.result()
        
        if news_results:
            titles = [res['title'] for res in news_results[:5]] # 상위 5개만 분석
            
            # 2. AI 모델(FinBERT)로 실제 분석
            analysis_results = sentiment_analyzer(titles)
            
            # 3. 분석 결과 출력 및 점수 계산
            total_sentiment_score = 0
            for i, (title, res) in enumerate(zip(titles, analysis_results)):
                label = res['label']
                score = res['score']
                
                # 점수화: positive(+1), negative(-1), neutral(0)
                val = 1 if label == "positive" else -1 if label == "negative" else 0
                total_sentiment_score += val
                
                color = "green" if label == "positive" else "red" if label == "negative" else "gray"
                st.markdown(f"**뉴스 {i+1}:** {title}")
                st.caption(f"결과: :{color}[{label.upper()} (신뢰도: {score:.2f})]")
            
            # 4. 최종 리포트
            avg_sentiment = total_sentiment_score / len(titles)
            if avg_sentiment > 0:
                st.success(f"현재 {selected_name}에 대한 뉴스는 대체로 **긍정적**입니다! (점수: {avg_sentiment:.2f})")
            elif avg_sentiment < 0:
                st.error(f"현재 {selected_name}에 대한 뉴스는 대체로 **부정적**입니다. (점수: {avg_sentiment:.2f})")
            else:
                st.info("뉴스가 중립적이거나 긍정/부정이 섞여 있습니다.")
        else:
            st.warning("최근 24시간 내에 검색된 뉴스가 없습니다.")