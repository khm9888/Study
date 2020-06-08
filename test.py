import pandas as pd 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC

#1. 데이터
iris = load_iris()
x = iris.data
y = iris.target
print(x.shape)  #(150, 4)
print(y.shape)  #(150, )
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=43)

# 그리드/랜덤 서치에서 사용할 매개 변수
parameters = [{"svm__C" : [1, 10, 100, 1000], "svm__kernel":['linear']}]
            #   {"svm__C" : [1, 10, 100, 1000], "svm__kernel":['rbf'], "svm__gamma" :[0.001, 0.0001]},
            #   {"svm__C" : [1, 10, 100, 1000], "svm__kernel":['sigmoid'], "svm__gamma" :[0.001, 0.0001]}] #20가지가 가능한 파라미터


#2. 모델
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

model = RandomizedSearchCV(pipe, parameters, cv=5)

#3. 훈련 
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print("최적의 매개 변수 : ", model.best_estimator_)
print("acc: ", acc)