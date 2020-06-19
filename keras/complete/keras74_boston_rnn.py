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

from sklearn.decomposition import PCA

pca = PCA(n_components=7)
x=pca.fit_transform(x)
print(f"x(before):{x}")
print(f"x.shape:{x.shape}")

#LSTM이니까 3차원
x=x.reshape(x.shape[0],x.shape[1],1)


#x,y값 나눔
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

#모델 구성 - 2차원(DNN)


from keras.models import Sequential
from keras.layers import Dense,LSTM

model = Sequential()

model.add(LSTM(1000,input_shape=(7,1),activation="relu",return_sequences=1))
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




import numpy as np

y_pre=y_pre.reshape(y_pre.shape[0])
print(__file__)
print(f"loss:{loss}")
print(f"rmse:{rmse(y_test,y_pre)}")
print(f"r2:{r2_score(y_test,y_pre)}")
print("-"*40)
print(f"y_test:{y_test}")
print(f"y_pre:{y_pre}")

#keras74_boston_rnn

"""
loss:26.8679182069977
rmse:5.183427461305385
r2:0.740185709087075
----------------------------------------
y_test:[38.7 24.7 25.  12.7 19.3 14.4 20.  19.6 12.3 29.6 50.  13.2 17.8 20.
 13.3 15.3 29.8  8.5 18.5 15.  17.4 37.9 11.3 29.1 29.  22.  19.  18.5
  7.2 25.  19.5 23.2 11.9 21.7 26.4 18.2 18.8 22.9 28.1 13.3 14.1 20.4
 50.  30.7 46.  15.4 21.2 21.4 24.4 16.4 50. ]
y_pre:[42.643505 32.365864 25.310463 16.67606  24.814182 20.133629 23.88747
 23.280048 14.781547 27.64526  42.185043 18.349632 19.17773  23.947807
 16.24144  20.661167 21.45716  11.172682 21.826454 24.183033 22.787624
 37.20114  14.854356 29.846344 32.343685 22.717144 14.902211 22.526627
 16.03979  25.828568 19.50077  26.503277 23.21318  25.744133 25.546371
 28.175423 21.018074 25.167492 24.961475 19.856758 14.503905 19.285126
 46.963577 36.892822 47.499912 21.605042 20.718842 24.589355 27.30943
 17.429768 33.59176 ]
"""