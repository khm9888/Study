import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense,Input

#데이터 입력

(x_train,y_train),(x_test,y_test) = mnist.load_data()#데이터 가져옴
x_train, x_test = x_train.reshape(-1,28*28),x_test.reshape(-1,28*28)#scale를 하기 위해 reshape


#데이터 전처리 -1

from keras.utils import np_utils

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)


from sklearn.preprocessing import MinMaxScaler
enc = MinMaxScaler()
x_train=enc.fit_transform(x_train)#scaler를 통해서 255로 나눔
x_test=enc.transform(x_test)

print(x_train.shape)


x = np.append(x_train,x_test,axis=0)

print(x.shape)

from sklearn.decomposition import PCA

# pca = PCA()
# pca.fit(x)
# cumsum=np.cumsum(pca.explained_variance_ratio_)
# print(cumsum)

# print(np.argmax(cumsum>=0.95)+1)#154

#데이터 전처리 -2

pca = PCA(n_components=154)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#모델구성

input1 = Input(shape=(154,))

dense = Dense(3000,activation="relu")(input1)
dense = Dense(10,activation="softmax")(dense)

model = Model(inputs=input1,outputs=dense)


model.summary()

#훈련

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])
model.fit(x_train,y_train,epochs=30,batch_size=256,validation_split=0.2)

#평가

loss,acc=model.evaluate(x_test,y_test,batch_size=256)
pred = model.predict(x_test)


print(acc)#0.9837999939918518

