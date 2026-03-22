FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# 이제 train.py 없이 app.py만 실행하면 끝!
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]