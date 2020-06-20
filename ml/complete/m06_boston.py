from sklearn.datasets import load_boston
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
# print(type(load_boston()))

values=load_boston()

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


# print("-"*30)
# print("KNeighborsClassifier")
# model = KNeighborsClassifier(1)

# #트레이닝

# # model.compile(loss="categorical_crossentropy",metircs=["acc"])
# model.fit(x_train,y_train)
# score = model.score(x_test,y_test)
# #test

# # loss,ac=model.evaluate(x_test,y_test)

# y_pre = model.predict(x_test)
# # print(f'r2:{r2_score(y_test,y_pre)}')

# print(f"score:{score}")
# print(f"acc:{acc(y_test,y_pre)}")
# print(f"y_test[0:20]:{y_test[0:20]}")
# print(f"y_pre[0:20]:{y_pre[0:20]}")



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
# print(f"acc:{acc(y_test,y_pre)}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")


# #모델구성-3
# print("-"*30)
# print("SVC")

# model3 = SVC()

# #트레이닝

# # model.compile(loss="categorical_crossentropy",metircs=["acc"])
# model3.fit(x_train,y_train)
# score = model3.score(x_test,y_test)
# #test

# # loss,ac=model.evaluate(x_test,y_test)

# y_pre = model3.predict(x_test)
# # print(f'r2:{r2_score(y_test,y_pre)}')

# print(f"score:{score}")
# print(f"r2:{r2_score(y_test,y_pre)}")
# print(f"acc:{acc(y_test,y_pre)}")
# print(f"y_test[0:20]:{y_test[0:20]}")
# print(f"y_pre[0:20]:{y_pre[0:20]}")


# #모델구성-4
# print("-"*30)
# print("LinearSVC")


# model4 = LinearSVC()

# #트레이닝

# # model.compile(loss="categorical_crossentropy",metircs=["acc"])
# model4.fit(x_train,y_train)
# score = model4.score(x_test,y_test)
# #test

# # loss,ac=model.evaluate(x_test,y_test)

# y_pre = model4.predict(x_test)
# # print(f'r2:{r2_score(y_test,y_pre)}')

# print(f"score:{score}")
# print(f"r2:{r2_score(y_test,y_pre)}")
# print(f"acc:{acc(y_test,y_pre)}")
# print(f"y_test[0:20]:{y_test[0:20]}")
# print(f"y_pre[0:20]:{y_pre[0:20]}")


# #모델구성-5
# print("-"*30)
# print("RandomForestClassifier")

# model5 = RandomForestClassifier()

# #트레이닝

# # model.compile(loss="categorical_crossentropy",metircs=["acc"])
# model5.fit(x_train,y_train)
# score = model5.score(x_test,y_test)
# #test

# # loss,ac=model.evaluate(x_test,y_test)

# y_pre = model5.predict(x_test)
# # print(f'r2:{r2_score(y_test,y_pre)}')

# print(f"score:{score}")
# print(f"r2:{r2_score(y_test,y_pre)}")
# print(f"acc:{acc(y_test,y_pre)}")
# print(f"y_test[0:20]:{y_test[0:20]}")
# print(f"y_pre[0:20]:{y_pre[0:20]}")


#모델구성-6
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
# print(f"acc:{acc(y_test,y_pre)}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

'''
------------------------------
KNeighborsRegressor
score:0.8663270056912061
r2:0.8663270056912061
y_test[0:20]:[19.5 18.2 21.4 18.9 24.7 20.6 24.8 50.  21.1 23.  19.1 18.5 18.6 17.4
 23.2 20.1 17.6 20.5 31.5 23.5]
y_pre[0:20]:[19.5 20.6 19.5 18.9 24.5 21.8 22.3 50.  20.7 21.5 19.9 18.7 13.5 19.1
 16.1 22.2 18.7 18.8 31.7 22.5]
------------------------------
RandomForestRegressor
score:0.8742737343712135
r2:0.8742737343712135
y_test[0:20]:[19.5 18.2 21.4 18.9 24.7 20.6 24.8 50.  21.1 23.  19.1 18.5 18.6 17.4
 23.2 20.1 17.6 20.5 31.5 23.5]
y_pre[0:20]:[18.846 19.548 21.037 20.699 23.722 21.478 24.083 38.482 20.446 22.48
 18.768 20.616 18.097 21.043 15.457 21.071 18.62  18.702 33.884 24.159]

'''