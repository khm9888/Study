import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist


#데이터 입력

(x_train,y_train),(x_test,y_test) = mnist.load_data()#데이터 가져옴

x_train, x_test = x_train.reshape(-1,28*28),x_test.reshape(-1,28*28)#scale를 하기 위해 reshape

from sklearn.preprocessing import MinMaxScaler

enc = MinMaxScaler()

x_train=enc.fit_transform(x_train)#scaler를 통해서 255로 나눔
x_test=enc.transform(x_test)

print(x_train.shape)

# from tensorflow.keras.utils import np_utils

# y_train=np_utils.to_categorical(y_train)
# y_test=np_utils.to_categorical(y_test)

#모델구성

x = np.append(x_train,x_test,axis=0)

print(x.shape)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(x)
cumsum=np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

print(np.argmax(cumsum>=0.95)+1)