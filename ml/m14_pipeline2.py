#iris, svc


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.layers import Dense, Input
from sklearn.metrics import r2_score,mean_squared_error as mse,accuracy_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC,SVC
from keras.utils import np_utils
from sklearn.datasets import load_iris
# from sklearn.ensemble import Grid

datasets = load_iris()

print(datasets.keys())

x= datasets.data
y= datasets.target

from sklearn.pipeline import Pipeline,make_pipeline
import  sklearn.pipeline

x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)

# print(RandomizedSearchCV.param_distributions())

# print(dir(Pipeline))

parameters = [
    {"svm__C":[1,10,100,1000],"svm__kernel":["linear"]}
    # {"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[0.001,0.0001]},
    # {"C":[1,10,100,1000],"kernel":["sigmoid"],"gamma":[0.001,0.0001]}
]

# print(1111111111111111,dir(sklearn.pipeline.make_pipeline()))

# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

pipe = make_pipeline(StandardScaler(), SVC())

# print(dir(pipe))
# print(pipe.__getattribute__)
model = RandomizedSearchCV(pipe,parameters,cv=5)

print(f"pipe.get_params():{pipe.get_params()}")

model.fit(x_train,y_train)

print(model.score(x_test,y_test))
y_pre = model.predict(x_test)

print(f"최적의 매개변수 = {model.best_estimator_}")
print(f"최종 정답률 = {accuracy_score(y_test,y_pre)}")
