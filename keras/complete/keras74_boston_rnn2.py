from sklearn.datasets import load_boston

#데이터 구성
dataset = load_boston()

x=dataset.data
y=dataset.target

#dimention 확인
print(f"x.shape:{x.shape}")
print(f"y.shape:{y.shape}")

print((x[0]))
print((y[0]))

#scaler 동작하기.

# from sklearn.preprocessing import RobustScaler
# scale = RobustScaler()
# x=scale.fit_transform(x)

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
x=scale.fit_transform(x)

# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# x=pca.fit_transform(x)
# print(f"x(before):{x}")
# print(f"x.shape:{x.shape}")

#LSTM이니까 3차원
x=x.reshape(x.shape[0],x.shape[1],1)


#x,y값 나눔
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

#모델 구성 - 2차원(DNN)


from keras.models import Sequential
from keras.layers import Dense,LSTM

model = Sequential()

model.add(LSTM(1000,input_shape=(13,1),activation="relu",return_sequences=1))
model.add(LSTM(1000,activation="relu"))
model.add(Dense(1,activation="relu"))

model.summary()

#트레이닝

model.compile(loss="mse", optimizer="adam")

model.fit(x_train,y_train,epochs=10,batch_size=1,validation_split=0.2)

#test

loss= model.evaluate(x_test,y_test,batch_size=1)

y_pre = model.predict(x_test)

from sklearn.metrics import mean_squared_error as mse , r2_score

def rmse(y_test,y_pre):
    return np.sqrt(mse(y_test,y_pre))

print(__file__)

print(f"loss:{loss}")

import numpy as np

y_pre=y_pre.reshape(y_pre.shape[0])

print(f"rmse:{rmse(y_test,y_pre)}")
print(f"r2:{r2_score(y_test,y_pre)}")
print("-"*40)
print(f"y_test:{y_test}")
print(f"y_pre:{y_pre}")

#keras74_boston_rnn

"""

"""