from sklearn.datasets import load_boston

#데이터 구성
dataset = load_boston()
x=dataset.data
y=dataset.target

print(dataset)

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


#x,y값 나눔
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

#모델 구성 - 2차원(DNN)


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(1000,input_shape=(7,),activation="relu"))
model.add(Dense(1,activation="relu"))

model.summary()

#트레이닝

model.compile(loss="mse", optimizer="adam", metrics=["mse","acc"])
model.fit(x_train,y_train,epochs=20,batch_size=1,validation_split=0.2)

#test

ev = model.evaluate(x_test,y_test,batch_size=1)
print(f"ev:{ev}")
y_pre = model.predict(x_test)

from sklearn.metrics import mean_squared_error as mse , r2_score

def rmse(y_test,y_pre):
    return np.sqrt(mse(y_test,y_pre))

print(__file__)

print(f"loss:{ev[0]}")
print(f"acc:{ev[2]}")

import numpy as np

y_pre=y_pre.reshape(y_pre.shape[0])

print(f"rmse:{rmse(y_test,y_pre)}")
print(f"r2:{r2_score(y_test,y_pre)}")

print(f"y_test:{y_test}")
print(f"y_pre:{y_pre}")


#keras73_boston_dnn
#epoch=20,n_component=0.74

"""
loss:27.65059175961177
acc:0.0
rmse:5.258382948237701
r2:0.7499346916402816
y_test:[29.1 15.6 42.3 20.6 13.9 21.7 18.5 19.3 18.6 19.6 20.6 32.9 23.2 28.1
  8.7  5.6 50.  22.  20.9 20.4 15.2 18.8 10.5 19.9 21.5 36.4 18.3  9.6
 46.7 35.2 15.4 29.  17.5 19.4 30.1 20.5 24.8 36.5  8.3 10.2 21.2 17.8
 43.8 18.4 13.5 20.4 32.  20.9 35.4 50.  22. ]
y_pre:[25.05929   17.192032  42.616375  23.928082  12.805841  21.89082
 19.666021  18.08603   22.368834  18.726046  21.263775  35.068905
 22.460861  21.979736   8.601236   6.8585863 19.311172  25.22787
 20.66204   20.973589  16.699738  19.255354   8.541396  18.369486
 16.73061   36.931675  16.96293   12.008586  37.414295  35.423378
 14.120199  34.27829   17.40825   19.491982  24.177132  22.170748
 30.41565   34.207508   8.7916765 10.209968  19.623669  21.523792
 38.716274  19.288012  12.735424  19.986893  29.678091  20.58783
 33.554695  41.03148   21.139704 ]
"""