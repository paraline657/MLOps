import yfinance as yf
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# 1. 삼성전자 주가 데이터 다운로드 (최근 1년)
print("삼성전자 데이터 다운로드 중...")
data = yf.download('005930.KS', period='1y')

# 2. 간단한 전처리 (내일의 종가를 예측하기 위해 데이터를 한 칸씩 밀어)
data['Target'] = data['Close'].shift(-1)
data = data.dropna()

# 특성(X): 시가, 고가, 저가, 종가, 거래량 / 타겟(y): 내일의 종가
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Target']

# 3. 모델 학습 (간단한 랜덤 포레스트)
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# 4. 모델 저장
joblib.dump(model, 'stock_model.pkl')
print("모델 학습 및 저장 완료!")