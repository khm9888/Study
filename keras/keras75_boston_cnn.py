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
x=x.reshape(x.shape[0],x.shape[1],1,1)
print(f"x.shape:{x.shape}")


#x,y값 나눔
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

#모델 구성 - 2차원(DNN)


from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten

model = Sequential()

model.add(Conv2D(40,(1,1),input_shape=(7,1,1),activation="relu"))
model.add(Conv2D(40,(1,1),activation="relu"))
model.add(Conv2D(40,(1,1),activation="relu"))
model.add(MaxPool2D(pool_size=1))
model.add(Flatten())
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
loss:28.928121376015685
rmse:5.3784869524099825
r2:0.6322006471082013
----------------------------------------
y_test:[48.8 18.5 18.5 19.6 33.  22.5 24.  35.2 26.7 32.4 22.6 36.2 18.5 23.1
 23.8 28.2 20.6 11.8 18.4 19.2 18.4 37.3 13.8 13.9 38.7 21.1 19.4 27.5
 22.1 15.6 35.1 23.2 20.6 15.6 21.2 20.4 20.8 20.   7.2 10.4 13.4 39.8
 19.9 33.1 13.1 19.8 24.8 43.8 23.3 20.2 36.1]
y_pre:[49.626278 18.709099 17.456678 19.925783 26.736698 20.60639  27.995935
 43.37379  34.785194 41.22325  20.281675 23.777554 21.319998 18.71523
 23.662115 26.789104 18.04341  16.544085 19.386492 20.017128 21.144596
 32.76241   8.312284 12.734607 40.988274 22.624655 17.850084 13.493588
 23.028486 17.692656 30.07714  20.637194 22.29194  13.729847 19.056479
 15.491387 20.262392 19.045012 17.617292 25.435987 11.535489 36.679348
 16.347403 41.48503  20.55291  17.235325 31.211275 38.1391   26.521772
 14.257256 30.24386 ]
"""