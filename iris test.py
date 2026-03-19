from sklearn.datasets import load_iris
# model_selsection -> model_selection으로 수정했어!
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. 데이터 준비
iris = load_iris()
x = iris.data
y = iris.target

# 2. 데이터 쪼개기 (학습용 80%, 시험용 20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print("학습할 자료 개수:", len(x_train))

# 3. 모델 만들고 학습시키기
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# 4. 정답 예측해보기
predictions = model.predict(x_test)

# 5. 정확도 확인
score = accuracy_score(y_test, predictions)
print(f"인공지능의 정답률: {score * 100}%")

joblib.dump(model, 'iris_model.pkl')