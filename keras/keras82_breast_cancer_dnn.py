from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler#scaler 처리
from keras.models import Model#모델
from keras.layers import Dense, Conv2D, MaxPool2D, Input, Flatten#모델
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
import pandas as pd

dataset=load_breast_cancer()


column=3
epoch=10
for column in range(5,14):
    x=dataset.data
    y=dataset.target

    # print(y[:])

    # df = pd.DataFrame(x, columns=dataset.feature_names)

    # print(df.head())


    # print(dataset.keys(),"\n",'-'*50)
    #dict_keys(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename']) 

    # [print (f"{i} : {dataset[i]} \n {'-'*50}") for i in list(dataset.keys())[2:-2]]

    # #dimention 확인
    # print(f"x.shape:{x.shape}")
    # print(f"y.shape:{y.shape}")

    # x.shape:(569, 30)
    # y.shape:(569,)

    # #1) 데이터 구성

    # print(f"x_test.shape:{x_test.shape}")
    # print(f"y_test.shape:{y_test.shape}")

    # #scaler 처리

    scaler = MinMaxScaler()

    x=scaler.fit_transform(x)


    # #y값에 대해서 to_categorical() 해주기
    y=np_utils.to_categorical(y)

    # print(f"y_test.shape:{y_test.shape}")

    # scaler를 통해서 나눔

    # #scaler 적용후 값 변화 확인 위해서
    # df = pd.DataFrame(x, columns=dataset.feature_names)
    # print(df.head())

    from sklearn.decomposition import PCA

    pca = PCA(n_components=column)
    x=pca.fit_transform(x)

    from sklearn.model_selection import train_test_split as tts
    x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

    # print(f"y{y}")


    #2) 모델

    input1= Input(shape=(column,))

    dense = Dense(10000,activation="relu")(input1)
    dense = Dense(2,activation="sigmoid")(dense)#이진분류

    model = Model(inputs=input1,outputs=dense)

    # model.summary()

    #3) 훈련

    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])
    model.fit(x_train,y_train,epochs=epoch,batch_size=1,validation_split=0.2,verbose=0)

    #4)테스트

    loss,acc = model.evaluate(x_test,y_test,batch_size=1)
    y_pre= model.predict(x_test)

    y_test=np.argmax(y_test, axis=-1)#인코딩 한 것에 대해 디코딩
    y_pre=np.argmax(y_pre, axis=-1)

    print("-"*40)
    # y_pre=y_pre.reshape(y_pre.shape[0])#print할 때 편히 보기 위해서 백터로 변환

    print(f"\nfile:{__file__}")
    print(f"n_component:{column}")
    print(f"epoch:{epoch}")
    print("="*50)
    print(f"loss:{loss}")
    print(f"acc:{acc}")
    print(f"y_test[0:20]:{y_test[0:20]}")
    print(f"y_pre[0:20]:{y_pre[0:20]}")
    
# keras82_boston_breast_cancer_dnn
'''
keras82_boston_breast_cancer_dnn
n_component:5
epoch:10
==================================================
loss:0.03172985937579674
acc:1.0
y_test[0:20]:[1 1 1 0 1 0 1 1 0 1 1 0 0 0 1 1 1 0 1 1]
y_pre[0:20]:[1 1 1 0 1 0 1 1 0 1 1 0 0 0 1 1 1 0 1 1]
57/57 [==============================] - 0s 1ms/step
----------------------------------------
keras82_boston_breast_cancer_dnn
n_component:6
epoch:10
==================================================
loss:0.11912173604812955
acc:0.9824561476707458
y_test[0:20]:[1 1 1 1 1 0 0 1 0 0 1 0 1 0 0 0 1 1 1 1]
y_pre[0:20]:[1 1 1 1 1 1 0 1 0 0 1 0 1 0 0 0 1 1 1 1]
57/57 [==============================] - 0s 1ms/step
----------------------------------------
keras82_boston_breast_cancer_dnn
n_component:7
epoch:10
==================================================
loss:0.698648758483253
acc:0.9298245906829834
y_test[0:20]:[0 0 0 0 0 0 1 0 1 0 0 1 1 0 0 1 1 1 1 0]
y_pre[0:20]:[0 0 0 1 0 0 1 0 1 0 0 1 1 0 0 1 1 1 1 0]
57/57 [==============================] - 0s 1ms/step
----------------------------------------
keras82_boston_breast_cancer_dnn
n_component:8
epoch:10
==================================================
loss:0.026857652831651105
acc:0.9824561476707458
y_test[0:20]:[0 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 0 1 1]
y_pre[0:20]:[0 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 0 1 1]
57/57 [==============================] - 0s 1ms/step
----------------------------------------
keras82_boston_breast_cancer_dnn
n_component:9
epoch:10
==================================================
loss:0.051583277881128826
acc:0.9824561476707458
y_test[0:20]:[1 0 1 1 1 1 1 0 1 0 0 1 1 0 0 1 0 1 0 1]
y_pre[0:20]:[1 0 1 1 1 1 1 0 1 0 0 1 0 0 0 1 0 1 0 1]
57/57 [==============================] - 0s 1ms/step
----------------------------------------
keras82_boston_breast_cancer_dnn
n_component:10
epoch:10
==================================================
loss:0.03502766053002588
acc:0.9824561476707458
y_test[0:20]:[0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 0]
y_pre[0:20]:[0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 0]
57/57 [==============================] - 0s 1ms/step
----------------------------------------
keras82_boston_breast_cancer_dnn
n_component:11
epoch:10
==================================================
loss:0.01367074967258283
acc:1.0
y_test[0:20]:[1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 0 1]
y_pre[0:20]:[1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 0 1]
57/57 [==============================] - 0s 1ms/step
----------------------------------------
keras82_boston_breast_cancer_dnn
n_component:12
epoch:10
==================================================
loss:0.06192707035274061
acc:0.9473684430122375
y_test[0:20]:[1 1 1 1 1 0 1 1 0 0 0 1 0 1 1 0 0 1 1 1]
y_pre[0:20]:[1 1 1 1 1 0 1 1 0 0 0 1 0 1 0 0 0 1 1 1]
57/57 [==============================] - 0s 1ms/step
----------------------------------------
keras82_boston_breast_cancer_dnn
n_component:13
epoch:10
==================================================
loss:0.1280237919900106
acc:0.9649122953414917
y_test[0:20]:[1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 0 0 1 0 1]
y_pre[0:20]:[1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 0 1 1 0 1]

'''