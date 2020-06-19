import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPool2D, Input,LSTM
from sklearn.datasets import load_diabetes
from keras.utils import np_utils
import pandas as pd

#데이터구성
dataset = load_diabetes()

# print(dataset.keys(),"\n",'-'*50)
# #dict_keys(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename']) 

# [print (f"{i} : {dataset[i]} \n {'-'*50}") for i in list(dataset.keys())[2:-2]]

# column=8
epoch=20

for column in range(4,8):


    x=dataset.data
    y=dataset.target

    df = pd.DataFrame(x, columns=dataset.feature_names)

    # print(df.head())

    # #dimention 확인
    # print(f"x.shape:{x.shape}")
    # print(f"y.shape:{y.shape}")
    # # x.shape:(442, 10)
    # # y.shape:(442,)


    # print(f"x[0]:{x[0]}")
    # print(f"y[0]:{y[0]}")

    # print(f"x:{x}")
    # print(f"y:{y}")



    # #y값이 다양한 것을 보고 다중분류로 판단했었음
    # from keras.utils import np_utils
    # y = np_utils.to_categorical(y)
    '''
    하지만, 혈중당 수치를 확인하는 걸로 보고, pca 적용후 회귀 분류 적용 예정
    '''



    # scaler를 통해서 255로 나눔
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x=scaler.fit_transform(x)

    # #scaler 적용후 값 변화 확인 위해서
    # df = pd.DataFrame(x, columns=dataset.feature_names)
    # print(df.head())

    from sklearn.decomposition import PCA

    pca = PCA(n_components=column)
    x=pca.fit_transform(x)

    #2->4 차원으로 변경

    x=x.reshape(-1,x.shape[1],1,1)


    from sklearn.model_selection import train_test_split as tts
    x_train,x_test,y_train,y_test = tts(x,y,train_size=0.9)

    # print(f"x_train[0]:{x_train[0]}")
    # print(f"y_train[0]:{y_train[0]}")


    #모델

    model = Sequential()

    model.add(Conv2D(40,(1,1),input_shape=(column,1,1),activation="relu"))
    model.add(Conv2D(40,(1,1),activation="relu"))
    model.add(Conv2D(40,(1,1),activation="relu"))
    model.add(MaxPool2D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(1,activation="relu"))#회귀모델

    # model.summary()

    #트레이닝

    model.compile(loss="mse", optimizer="adam")
    model.fit(x_train,y_train,batch_size=1,epochs=epoch,validation_split=0.3,verbose=1)

    #테스트

    from sklearn.metrics import mean_squared_error as mse , r2_score

    def rmse(y_test,y_pre):
        return np.sqrt(mse(y_test,y_pre))

    print("-"*40)

    loss = model.evaluate(x_test,y_test,batch_size=1)

    y_pre=model.predict(x_test)
    y_pre=y_pre.reshape(y_pre.shape[0])#print할 때 편히 보기 위해서 백터로 변환

    print(__file__)
    print(f"n_component:{column}")
    print(f"epoch:{epoch}")
    print("="*50)

    print(f"loss:{loss}")
    print(f"rmse:{rmse(y_test,y_pre)}")
    print(f"r2:{r2_score(y_test,y_pre)}")
    print(f"y_test[0:20]:{y_test[0:20]}")
    print(f"y_pre[0:20]:{y_pre[0:20]}")

    #keras81_boston_diabets_cnn

'''
keras81_boston_diabets_cnn
n_component:4
epoch:20
==================================================
loss:2708.494905339347
rmse:52.04320208645594
r2:0.4506020727692104
y_test[0:20]:[127. 190.  78. 270. 182. 180.  97. 168. 201. 202. 151. 128. 202. 142.
 303. 184. 198. 144.  78. 135.]
y_pre[0:20]:[169.84644  200.20947  119.52948  252.8861   125.22897  187.86728
  99.2404   106.00895   98.745636 143.43513  192.92726   94.201385
 125.74633  144.84229  270.98248  149.72328  212.03052  169.76999
  90.9175   133.955   ]


keras81_boston_diabets_cnn
n_component:5
epoch:20
==================================================
loss:2114.428151798248
rmse:45.98291163493198
r2:0.6030656050291282
y_test[0:20]:[ 71. 236.  85. 104.  49. 202.  47. 178.  84.  53. 261. 208. 230.  87.
 144. 262.  63. 158.  91. 111.]
y_pre[0:20]:[121.14325  220.49507  128.54448   80.62222  101.55207  188.80087
  59.74207  189.03934  110.537605 106.47482  243.78937  224.97537
 142.45525   65.90435  166.93404  165.02802   62.031414  61.115753
 183.08678  117.19015 ]
 
 keras81_boston_diabets_cnn
n_component:6
epoch:20
==================================================
loss:28831.11111111111
rmse:169.79726473389113
r2:-4.118763591907333
y_test[0:20]:[128. 142. 258. 201. 310. 116. 170.  52. 217. 200.  72.  48. 132. 187.
 161.  85. 121. 160. 142. 275.]
y_pre[0:20]:[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

keras81_boston_diabets_cnn
n_component:7
epoch:20
==================================================
loss:3292.9365815520287
rmse:57.384114483408545
r2:0.28884145311749765
y_test[0:20]:[ 72. 272. 199.  97.  71. 275. 242. 162. 113. 230. 101.  87. 154.  97.
  90. 131. 225. 179. 200. 206.]
y_pre[0:20]:[ 54.272526 220.07408  107.35569  111.29249  121.935425 226.2739
 289.35858  106.79861  112.04814  140.29756  116.69268   58.34475
 151.62892  138.80707   63.519115 201.21257  207.67784  104.03074
 125.46774  152.10933 ]
'''