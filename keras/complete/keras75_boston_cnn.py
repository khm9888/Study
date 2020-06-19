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
keras80_boston_diabets_rnn
n_component:4
epoch:30
==================================================
loss:2717.6101504325866
rmse:52.13070248322287
r2:0.5360334167327407
y_test[0:20]:[281. 161. 118. 191. 195.  75. 233.  71. 221.  64. 200.  63. 150.  96.
  78. 252.  44.  79.  80. 143.]
y_pre[0:20]:[275.82806  175.21278   94.10723  121.950714 231.16481   66.23316
 249.92671  131.17378  170.8031   102.58203  110.415344  59.1734
 214.74626   86.10005  188.12395  153.1397   130.09239  123.09619
  99.15425   93.56562 ]
----------------------------------------
45/45 [==============================] - 0s 2ms/step
keras80_boston_diabets_rnn
n_component:5
epoch:30
==================================================
loss:4036.5049808714125
rmse:63.53349473523536
r2:0.3192122934754066
y_test[0:20]:[249. 115.  96. 180. 268. 272.  77. 104. 139.  25. 134. 113.  72.  90.
 102. 180. 275. 138.  65. 220.]
y_pre[0:20]:[226.95935   94.49394   92.98869  187.99022  240.14145  211.7979
 198.96301   95.144554 200.78174  152.87393   86.90535  111.0022
 128.01288   79.85974  137.3521   243.20439  221.27824  197.03996
 100.27869  211.29326 ]

----------------------------------------
45/45 [==============================] - 0s 2ms/step
keras80_boston_diabets_rnn
n_component:8
epoch:30
==================================================
loss:3837.528462960985
rmse:61.94778843039411
r2:0.4391979179284903
y_test[0:20]:[181. 121.  40.  77. 268. 306. 202. 109. 216. 186.  88.  78. 100.  75.
  69.  84.  39. 302. 139. 257.]
y_pre[0:20]:[165.30447  218.19087  146.62465   97.15219  213.44618  248.23216
 149.76703  214.10907  190.10161  193.02905  107.008865  86.64221
 146.76787   70.393234  96.15564  206.91129   71.85997  141.91985
 121.17405  188.8994  ]
PS D:\Study>
"""