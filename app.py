import streamlit as st
import joblib
import numpy as np

# 1. 저장했던 인공지능 '뇌' 불러오기
model = joblib.load('iris_model.pkl')

# 2. 웹 화면 꾸미기
st.title("🌸 붓꽃 종류 맞추기 인공지능")
st.write("꽃잎과 꽃받침의 길이를 입력하면 인공지능이 종류를 맞춰줍니다!")

# 3. 사용자로부터 입력 받기 (슬라이더 바)
sepal_l = st.slider("꽃받침 길이 (Sepal Length)", 4.0, 8.0, 5.0)
sepal_w = st.slider("꽃받침 넓이 (Sepal Width)", 2.0, 4.5, 3.0)
petal_l = st.slider("꽃잎 길이 (Petal Length)", 1.0, 7.0, 4.0)
petal_w = st.slider("꽃잎 넓이 (Petal Width)", 0.1, 2.5, 1.0)

# 4. 버튼을 누르면 예측 시작!
if st.button("어떤 꽃일까?"):
    # 입력받은 데이터를 모델이 이해할 수 있는 형태로 변환
    input_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    prediction = model.predict(input_data)
    
    # 정답 숫자를 이름으로 바꿔주기
    flower_names = ['Setosa', 'Versicolor', 'Virginica']
    result = flower_names[prediction[0]]
    
    # 결과 화면에 출력
    st.success(f"이 꽃은 바로... **{result}** 입니다!")
