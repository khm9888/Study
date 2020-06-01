from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler#scaler 처리
from keras.models import Model,Sequential#모델
from keras.layers import Dense, Conv2D, MaxPool2D, Input, Flatten,LSTM#모델
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
import pandas as pd

dataset=load_breast_cancer()

print(type(dataset))

print(dataset.keys())


df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df.head())
# column=3
# epoch=30
# for column in range(5,7):
#     x=dataset.data
#     y=dataset.target

#     # print(y[:])

#     # df = pd.DataFrame(x, columns=dataset.feature_names)

#     # print(df.head())


#     # print(dataset.keys(),"\n",'-'*50)
#     #dict_keys(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename']) 

#     # [print (f"{i} : {dataset[i]} \n {'-'*50}") for i in list(dataset.keys())[2:-2]]

#     # #dimention 확인
#     # print(f"x.shape:{x.shape}")
#     # print(f"y.shape:{y.shape}")

#     # x.shape:(569, 30)
#     # y.shape:(569,)

#     # #1) 데이터 구성

#     # print(f"x_test.shape:{x_test.shape}")
#     # print(f"y_test.shape:{y_test.shape}")

#     # #scaler 처리

#     scaler = MinMaxScaler()

#     x=scaler.fit_transform(x)


#     # #y값에 대해서 to_categorical() 해주기
#     y=np_utils.to_categorical(y)

#     # print(f"y_test.shape:{y_test.shape}")

#     # scaler를 통해서 나눔

#     # #scaler 적용후 값 변화 확인 위해서
#     # df = pd.DataFrame(x, columns=dataset.feature_names)
#     # print(df.head())

#     from sklearn.decomposition import PCA

#     pca = PCA(n_components=column)
#     x=pca.fit_transform(x)
    
#     #2->3 차원으로 변경

#     x=x.reshape(-1,x.shape[1],1,1)

#     from sklearn.model_selection import train_test_split as tts
#     x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

#     # print(f"y{y}")


#     #2) 모델
#     model = Sequential()

#     model.add(Conv2D(40,(1,1),input_shape=(column,1,1),activation="relu"))
#     model.add(Conv2D(40,(1,1),activation="relu"))
#     model.add(Conv2D(40,(1,1),activation="relu"))
#     model.add(MaxPool2D(pool_size=1))
#     model.add(Flatten())
#     model.add(Dense(2,activation="sigmoid"))#이진분류
#     # model.summary()

#     #3) 훈련

#     model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])
#     model.fit(x_train,y_train,epochs=epoch,batch_size=1,validation_split=0.2,verbose=2)

#     #4)테스트

#     loss,acc = model.evaluate(x_test,y_test,batch_size=1)
#     y_pre= model.predict(x_test)

#     y_test=np.argmax(y_test, axis=-1)#인코딩 한 것에 대해 디코딩
#     y_pre=np.argmax(y_pre, axis=-1)

#     print("-"*40)
#     # y_pre=y_pre.reshape(y_pre.shape[0])#print할 때 편히 보기 위해서 백터로 변환

#     print(f"\n{__file__}")
#     print(f"n_component:{column}")
#     print(f"epoch:{epoch}")
#     print("="*50)
#     print(f"loss:{loss}")
#     print(f"acc:{acc}")
#     print(f"y_test[0:20]:{y_test[0:20]}")
#     print(f"y_pre[0:20]:{y_pre[0:20]}")
    
# # keras84_boston_breast_cancer_cnn
# '''
# keras84_boston_breast_cancer_cnn
# n_component:5
# epoch:30
# ==================================================
# loss:0.2949465449138593
# acc:0.9473684430122375
# y_test[0:20]:[1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 0 1 1 0 1]
# y_pre[0:20]:[1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 0 1 1 0 1]
# ----------------------------------------
# keras84_boston_breast_cancer_cnn
# n_component:6
# epoch:30
# ==================================================
# loss:0.007088758578284934
# acc:1.0
# y_test[0:20]:[0 0 1 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1]
# y_pre[0:20]:[0 0 1 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1]

# '''