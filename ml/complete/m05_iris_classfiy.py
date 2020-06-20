from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error as mse,accuracy_score as acc
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import LinearSVC,SVC


# from keras.utils import np_utils
import pandas as pd

#데이터입력
# print(type(load_iris()))

values=load_iris()

print(values.keys())

x=values.data
y=values.target

# print(y)
# df = pd.DataFrame()
# y=

x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)

# x_train

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x.shape)

#모델구성

# input1 = Input(shape=(4,))
# dense1 = Dense(1000,activation="relu")(input1)
# dense1 = Dense(1000,activation="softmax")(dense1)
# model = Model(inputs = input1, outputs = dense1)


print("-"*30)
print("KNeighborsClassifier")
model = KNeighborsClassifier(1)

#트레이닝

# model.compile(loss="categorical_crossentropy",metircs=["acc"])
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
#test

# loss,ac=model.evaluate(x_test,y_test)

y_pre = model.predict(x_test)
# print(f'r2:{r2_score(y_test,y_pre)}')

print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc(y_test,y_pre)}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")



#모델구성-2
print("-"*30)
print("KNeighborsRegressor")

model2 = KNeighborsRegressor(1)

#트레이닝

# model.compile(loss="categorical_crossentropy",metircs=["acc"])
model2.fit(x_train,y_train)
score = model2.score(x_test,y_test)
#test

# loss,ac=model.evaluate(x_test,y_test)

y_pre = model2.predict(x_test)
# print(f'r2:{r2_score(y_test,y_pre)}')

print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc(y_test,y_pre)}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")


#모델구성-3
print("-"*30)
print("SVC")

model3 = SVC()

#트레이닝

# model.compile(loss="categorical_crossentropy",metircs=["acc"])
model3.fit(x_train,y_train)
score = model3.score(x_test,y_test)
#test

# loss,ac=model.evaluate(x_test,y_test)

y_pre = model3.predict(x_test)
# print(f'r2:{r2_score(y_test,y_pre)}')

print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc(y_test,y_pre)}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

#모델구성-4
print("-"*30)
print("LinearSVC")


model4 = LinearSVC()

#트레이닝

# model.compile(loss="categorical_crossentropy",metircs=["acc"])
model4.fit(x_train,y_train)
score = model4.score(x_test,y_test)
#test

# loss,ac=model.evaluate(x_test,y_test)

y_pre = model4.predict(x_test)
# print(f'r2:{r2_score(y_test,y_pre)}')

print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc(y_test,y_pre)}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

#모델구성-5
print("-"*30)
print("RandomForestClassifier")

model5 = RandomForestClassifier()

#트레이닝

# model.compile(loss="categorical_crossentropy",metircs=["acc"])
model5.fit(x_train,y_train)
score = model5.score(x_test,y_test)
#test

# loss,ac=model.evaluate(x_test,y_test)

y_pre = model5.predict(x_test)
# print(f'r2:{r2_score(y_test,y_pre)}')

print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc(y_test,y_pre)}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")


#모델구성-5
print("-"*30)
print("RandomForestRegressor")

model6 = RandomForestRegressor()

#트레이닝

# model.compile(loss="categorical_crossentropy",metircs=["acc"])
model6.fit(x_train,y_train)
score = model6.score(x_test,y_test)
#test

# loss,ac=model.evaluate(x_test,y_test)

y_pre = model6.predict(x_test)
# print(f'r2:{r2_score(y_test,y_pre)}')

print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc(y_test,y_pre)}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

'''
------------------------------
KNeighborsClassifier
score:1.0
acc:1.0
y_test[0:20]:[1 1 2 0 0 1 1 0 0 1 2 1 2 1 0 1 2 0 1 1]
y_pre[0:20]:[1 1 2 0 0 1 1 0 0 1 2 1 2 1 0 1 2 0 1 1]
------------------------------
KNeighborsRegressor
score:1.0
r2:1.0
acc:1.0
y_test[0:20]:[1 1 2 0 0 1 1 0 0 1 2 1 2 1 0 1 2 0 1 1]
y_pre[0:20]:[1. 1. 2. 0. 0. 1. 1. 0. 0. 1. 2. 1. 2. 1. 0. 1. 2. 0. 1. 1.]
------------------------------
SVC
score:1.0
r2:1.0
acc:1.0
y_test[0:20]:[1 1 2 0 0 1 1 0 0 1 2 1 2 1 0 1 2 0 1 1]
y_pre[0:20]:[1 1 2 0 0 1 1 0 0 1 2 1 2 1 0 1 2 0 1 1]
------------------------------
LinearSVC
score:0.9333333333333333
r2:0.8809523809523809
acc:0.9333333333333333
y_test[0:20]:[1 1 2 0 0 1 1 0 0 1 2 1 2 1 0 1 2 0 1 1]
y_pre[0:20]:[1 1 2 0 0 1 1 0 0 1 2 1 2 2 0 1 2 0 1 1]
------------------------------
RandomForestClassifier
score:1.0
r2:1.0
acc:1.0
y_test[0:20]:[1 1 2 0 0 1 1 0 0 1 2 1 2 1 0 1 2 0 1 1]
y_pre[0:20]:[1 1 2 0 0 1 1 0 0 1 2 1 2 1 0 1 2 0 1 1]
------------------------------
RandomForestRegressor
score:0.9998214285714285
r2:0.9998214285714285
y_test[0:20]:[1 1 2 0 0 1 1 0 0 1 2 1 2 1 0 1 2 0 1 1]
y_pre[0:20]:[1.   1.   2.   0.   0.   1.   1.   0.   0.   1.   1.99 1.   2.   1.04
 0.   1.   1.98 0.   1.   1.03]

'''