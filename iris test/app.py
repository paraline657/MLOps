import streamlit as st
import joblib
import numpy as np
from groq import Groq
import os

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'iris_model.pkl')

# 1. 저장했던 인공지능 '뇌' 불러오기
model = joblib.load(model_path)

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

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.divider() # 화면에 줄 긋기
st.subheader("🤖 붓꽃 전문가 챗봇")

# 2. 채팅 메시지를 저장할 공간 만들기 (Streamlit 특성상 필요해)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 이전에 나눈 대화 화면에 보여주기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. 사용자 입력 받기
if prompt := st.chat_input("붓꽃에 대해 궁금한 점을 물어보세요!"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        # 호출 방식이 OpenAI랑 거의 똑같아서 쉬워!
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[
                {"role": "system", "content": "너는 붓꽃 전문가야."},
                {"role": "user", "content": prompt}
            ]
        )
        full_response = response.choices[0].message.content
        st.markdown(full_response)
        
    # 답변도 저장하기
    st.session_state.messages.append({"role": "assistant", "content": full_response})