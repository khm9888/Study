#wine 퀄 classify 70 가장 우측이 wine quality


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from sklearn.metrics import r2_score,mean_squared_error as mse,accuracy_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC,SVC
from keras.utils import np_utils
# from sklearn.preprocessing import OneHotEncoder
#0. 데이터 전처리

wine_df = pd.read_csv("./data/winequality-white.csv",header=0,encoding="cp949",sep=";")


# print(wine_df.info())
# print(wine_df.columns)
print(wine_df.head)

x = wine_df.iloc[:,:-1].values
y = wine_df.iloc[:,-1].values

print(x)
print(y)

print(f"x.shape:{x.shape}")
print(f"y.shape:{y.shape}")

#1. 데이터 입력

x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)

# scale
# 머신러닝이니까 스킵
# scale = StandardScaler()
# x_train = scale.fit_transform(x_train)
# x_test = scale.transform(x_test)

#다중분류니까, to_categorical
#하지만 머신러닝이라서 하지 않는다.

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

#2. 모델구성-1

print("-"*30+"SVC"+"-"*30)

model = SVC()

#3.훈련

model.fit(x_train,y_train)

#4.평가 및 예측

y_pre = model.predict(x_test)

score = model.score(x_test,y_test)
acc = accuracy_score(y_test,y_pre)


print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

print("-"*60)

#2. 모델구성-2

print("-"*30+"LinearSVC"+"-"*30)

model = LinearSVC()

#3.훈련

model.fit(x_train,y_train)

#4.평가 및 예측

y_pre = model.predict(x_test)

score = model.score(x_test,y_test)
acc = accuracy_score(y_test,y_pre)


print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

print("-"*60)

#2. 모델구성-3

print("-"*30+"KNeighborsClassifier"+"-"*30)

model = KNeighborsClassifier()

#3.훈련

model.fit(x_train,y_train)

#4.평가 및 예측

y_pre = model.predict(x_test)

score = model.score(x_test,y_test)
acc = accuracy_score(y_test,y_pre)


print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

print("-"*60)

#2. 모델구성-4

print("-"*30+"KNeighborsRegressor"+"-"*30)

model = KNeighborsRegressor()

#3.훈련

model.fit(x_train,y_train)

#4.평가 및 예측

y_pre = model.predict(x_test)

score = model.score(x_test,y_test)
# acc = accuracy_score(y_test,y_pre)


print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
# print(f"acc:{acc}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

print("-"*60)

#2. 모델구성-5

print("-"*30+"RandomForestClassifier"+"-"*30)

model = RandomForestClassifier()

#3.훈련

model.fit(x_train,y_train)

#4.평가 및 예측

y_pre = model.predict(x_test)

score = model.score(x_test,y_test)
acc = accuracy_score(y_test,y_pre)


print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

print("-"*60)

#2. 모델구성-6

print("-"*30+"RandomForestRegressor"+"-"*30)

model = RandomForestRegressor()

#3.훈련

model.fit(x_train,y_train)

#4.평가 및 예측

y_pre = model.predict(x_test)

score = model.score(x_test,y_test)
# acc = accuracy_score(y_test,y_pre)


print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
# print(f"acc:{acc}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

print("-"*60)
'''
------------------------------SVC------------------------------
score:0.5785714285714286
r2:0.26449696846235904
acc:0.5785714285714286
y_test[0:20]:[6 7 6 6 6 6 8 6 5 6 5 5 5 6 5 7 5 6 7 6]
y_pre[0:20]:[6 6 6 6 6 6 7 6 5 6 5 5 6 6 5 6 5 6 6 6]
------------------------------------------------------------

------------------------------LinearSVC------------------------------
score:0.5193877551020408
r2:0.10688917599000747
acc:0.5193877551020408
y_test[0:20]:[6 7 6 6 6 6 8 6 5 6 5 5 5 6 5 7 5 6 7 6]
y_pre[0:20]:[6 6 6 6 6 5 6 6 5 6 6 5 5 6 5 6 5 6 6 6]
------------------------------------------------------------

------------------------------KNeighborsClassifier------------------------------
score:0.5612244897959183
r2:0.10688917599000747
acc:0.5612244897959183
y_test[0:20]:[6 7 6 6 6 6 8 6 5 6 5 5 5 6 5 7 5 6 7 6]
y_pre[0:20]:[6 6 6 6 6 6 7 6 6 6 5 5 6 6 5 6 5 6 6 6]
------------------------------------------------------------

------------------------------KNeighborsRegressor------------------------------
score:0.3515490058379213
r2:0.35154900583792137
y_test[0:20]:[6 7 6 6 6 6 8 6 5 6 5 5 5 6 5 7 5 6 7 6]
y_pre[0:20]:[5.6 5.6 5.8 6.  5.6 5.8 6.6 5.6 5.6 6.  5.2 5.2 6.  6.2 5.6 6.6 5.  5.6
 6.4 5.6]
------------------------------------------------------------

------------------------------RandomForestClassifier------------------------------
score:0.6938775510204082
r2:0.45231292115857813
acc:0.6938775510204082
y_test[0:20]:[6 7 6 6 6 6 8 6 5 6 5 5 5 6 5 7 5 6 7 6]
y_pre[0:20]:[6 6 6 6 6 6 7 6 5 6 6 5 6 6 5 6 5 6 6 6]
------------------------------------------------------------

------------------------------RandomForestRegressor------------------------------
score:0.5402006738537249
r2:0.5402006738537249
y_test[0:20]:[6 7 6 6 6 6 8 6 5 6 5 5 5 6 5 7 5 6 7 6]
y_pre[0:20]:[5.99 6.16 5.93 6.07 5.56 5.88 6.71 5.67 5.07 6.63 5.54 5.1  5.96 6.21
 5.21 6.6  5.11 5.96 6.61 6.04]
------------------------------------------------------------


'''