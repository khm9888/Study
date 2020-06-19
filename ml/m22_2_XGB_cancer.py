# 과적합 방지
# 1. 훈련 데이터양을 늘린다.
# 2. 피처수를 줄인다.
# 3. regularization

from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import matplotlib.pyplot as plt

# 회기 모델
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)


# 랜포, 트리 -> 업그레이드 -> 부스트
# 트리의 특징 전처리가 필요없음 , 트리의 특징에도? 결측치제거가 필요없음? 부스트의 특징아닌가욤?
# 앙상블 모델이라 속도가 느림 

# #  로스와 옵티마이저에 관련이 있음  xgb 파라미터
# n_estimators = 210 # 나무의 숫자
# learning_rate = 0.035 # 학습률? 디폴트 0.01임 노드 건드리는 것보다 이걸 수정하는게 더 좋을수 있음 핵심키워드 
# # 좌우 상하로 dropout
# colsample_bytree = 0.745 # 우승모델은 0.6 ~ 0.9 라고 하는데 참고만 하자 디폴트는 1임 dropout
# colsample_bylevel = 0.73 # 우승모델은 0.6 ~ 0.9 라고 하는데 참고만 하자 디폴트는 1임  dropout

#  로스와 옵티마이저에 관련이 있음  xgb 파라미터
n_estimators = 220 # 나무의 숫자
learning_rate = 0.035 # 학습률? 디폴트 0.01임 노드 건드리는 것보다 이걸 수정하는게 더 좋을수 있음 핵심키워드 
# 좌우 상하로 dropout
colsample_bytree = 0.75 # 우승모델은 0.6 ~ 0.9 라고 하는데 참고만 하자 디폴트는 1임 dropout
colsample_bylevel = 0.99 # 우승모델은 0.6 ~ 0.9 라고 하는데 참고만 하자 디폴트는 1임  dropout

'''learning rate, bytree 를 변화했을때 달라지는 폭이 큼 다른거에 비해

    가중치를 조절하는 방법? '''

max_depth = 4 # 어디다가 주석 달았는데 기억이 안남 ㅋ -> 나무의 깊이 디폴트는 3인가 5인가 정도라고함 
n_jobs = -1

# parameters = {"n_estimators": [],
#               "max_depth": [4, 8, 12, 16],
#               "max_features": [3, 5, 7, 9],
#               "min_samples_split": [3, 5, 7, 9]}

model = XGBClassifier(  n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        colsample_bytree=colsample_bytree,
                        # colsample_bylevel=colsample_bylevel,
                        max_depth=max_depth,
                        n_jobs=n_jobs   )

model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print(f"score : {score}")


print(f"feature importance : {model.feature_importances_}")

plot_importance(model)
plt.show()