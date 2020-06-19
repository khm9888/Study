#과적합 방지
#1.훈련데이터 늘린다.
#2.feature를 줄인다.
#3.regularization

from xgboost import XGBRegressor,plot_importance
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

dataset = load_boston()
# print(type(load_boston())) #<class 'sklearn.utils.Bunch'>
# print(print(dataset.keys()))

x= dataset.data
y= dataset.target

print(type(x)) #<class 'numpy.ndarray'>

# print(x.shape)

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test  = tts(x,y,train_size=0.8)

n_estimators = 100
learning_rate = 0.01
colsample_bytree=0.9#우승모델은 0.6~0.9
colsample_bylevel=0.9#우승모델은 0.6~0.9

max_depth = 5
n_jobs=-1


model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, colsample_bytree=colsample_bytree,
                      colsample_bylevel=colsample_bylevel,n_jobs=n_jobs)

#트리구조는 전처리를 안 해도 됨. - pipeline,
#결측치 제거 안 해도 됨 - 보간법 적용.
#해주긴 하지만, 보간법 적용하는게 나을 수 있음.
#여러모델보단 느림. 앙상블이라.

model.fit(x_train,y_train)

r2=model.score(x_test,y_test)

print(model.feature_importances_)
f_i = model.feature_importances_

plot_importance(model)
plt.show()



