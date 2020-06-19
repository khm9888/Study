#wine 퀄 classify 70 가장 우측이 wine quality
#wine 퀄 classify 70 가장 우측이 wine quality


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import Dense, Input,LSTM
from sklearn.metrics import r2_score,mean_squared_error as mse,accuracy_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC,SVC
from keras.utils import np_utils
from sklearn.decomposition import PCA
# from sklearn.preprocessing import OneHotEncoder


for pca_n in range(8,9):
    
    #0. 데이터 전처리

    wine_df = pd.read_csv("./data/winequality-white.csv",header=0,encoding="cp949",sep=";")

    # print(wine_df.info())
    # print(wine_df.columns)
    # print(wine_df.head)
    
    x = wine_df.iloc[:,:-1].values
    y = wine_df.iloc[:,-1].values

    # print(x)
    # print(y)

    # print(f"x.shape:{x.shape}")
    # print(f"y.shape:{y.shape}")

    #1. 데이터 입력

    x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)

    # scale
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)

    #PCA

    pca = PCA(pca_n)
    x_train=pca.fit_transform(x_train)
    x_test=pca.transform(x_test)


    #다중분류니까, to_categorical

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # print(f"x_train.shape:{x_train.shape}")
    # print(f"y_train.shape:{y_train.shape}")
    
    
    
    

    #2. 모델구성

    model = Sequential()

    model.add(Dense(1000,input_shape=(pca_n,),activation="relu"))
    model.add(Dense(10,activation="softmax"))#다중분류니까

    #3.훈련

    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])
    model.fit(x_train,y_train,batch_size=1,epochs=10,validation_split=0.3)

    #4.평가 및 예측

    y_pre = model.predict(x_test)

    # score = model.score(x_test,y_test)



    y_test=np.argmax(y_test,axis=1)
    y_pre=np.argmax(y_pre,axis=1)

    acc = accuracy_score(y_test,y_pre)



    # print(f"score:{score}")
    print(f"pca_n:{pca_n}")
    print(f"r2:{r2_score(y_test,y_pre)}")
    print(f"acc:{acc}")
    print(f"y_test[0:20]:{y_test[0:20]}")
    print(f"y_pre[0:20]:{y_pre[0:20]}")

    print("-"*60)


#-------------------------------------------------------------------------------------------------------------------------
#lstm
    
    #0. 데이터 전처리

    wine_df = pd.read_csv("./data/winequality-white.csv",header=0,encoding="cp949",sep=";")


    # print(wine_df.info())
    # print(wine_df.columns)
    # print(wine_df.head)
    
    x = wine_df.iloc[:,:-1].values
    y = wine_df.iloc[:,-1].values

    # print(x)
    # print(y)

    # print(f"x.shape:{x.shape}")
    # print(f"y.shape:{y.shape}")

    #1. 데이터 입력

    x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)

    # scale
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)

    #PCA

    pca = PCA(pca_n)
    x_train=pca.fit_transform(x_train)
    x_test=pca.transform(x_test)


    #다중분류니까, to_categorical

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # print(f"x_train.shape:{x_train.shape}")
    # print(f"y_train.shape:{y_train.shape}")
    
    
    #LSTM, reshape
    x_train = x_train.reshape(-1,x_train.shape[1],1)
    x_test = x_test.reshape(-1,x_test.shape[1],1)

    #2. 모델구성

    model = Sequential()

    model.add(LSTM(1000,input_shape=(pca_n,1),activation="relu"))
    model.add(Dense(10,activation="softmax"))#다중분류니까

    #3.훈련

    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])
    model.fit(x_train,y_train,batch_size=1,epochs=10,validation_split=0.3)

    #4.평가 및 예측

    y_pre = model.predict(x_test)

    # score = model.score(x_test,y_test)



    y_test=np.argmax(y_test,axis=1)
    y_pre=np.argmax(y_pre,axis=1)

    acc = accuracy_score(y_test,y_pre)



    # print(f"score:{score}")
    print(f"pca_n:{pca_n}")
    print(f"r2:{r2_score(y_test,y_pre)}")
    print(f"acc:{acc}")
    print(f"y_test[0:20]:{y_test[0:20]}")
    print(f"y_pre[0:20]:{y_pre[0:20]}")

    print("-"*60)

