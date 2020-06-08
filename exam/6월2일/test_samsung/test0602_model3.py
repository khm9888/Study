import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from keras.models import Model
from keras.layers import Dense, Input, Concatenate
from sklearn.metrics import r2_score,mean_squared_error as mse
from keras.callbacks import EarlyStopping


samsung = pd.read_csv("./data/csv/samsung.csv",index_col=0,header=0,encoding="cp949",sep=",")
hite = pd.read_csv("./data/csv/hite.csv",index_col=0,header=0,encoding="cp949")

samsung = samsung.dropna(axis=0)#pandas에서는 행import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from keras.models import Sequential, Model



def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)

size = 6

# 1. 데이터
# npy 불러오기
samsung = np.load('./data/samsung.npy', allow_pickle = 'True') # 객체 배열(object)를 저장할 수 있게 해줌
hite = np.load('./data/hite.npy', allow_pickle = 'True') # object는 str, int와 같은 데이터 타입

print(samsung.shape)        # (509, 1)
print(hite.shape)           # (509, 5)

samsung = samsung.reshape(samsung.shape[0], )
print(samsung.shape)        # (509, )

samsung = split_x(samsung, size)
print(samsung.shape)        # (504, 6)

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

print(x_sam.shape)          # (504, 5)
print(y_sam.shape)          # (504, )

x_hit = hite[5:510, :]
print(x_hit.shape)          # (504, 5)

x_hit = x_hit.reshape(x_hit.shape[0], 5, 1)
print(x_hit.shape)          # (504, 5, 1)

x_sam = x_sam.reshape(x_sam.shape[0], 5, 1)
print(x_sam.shape)          # (504, 5, 1)


#2. 모델구성 
input1 = Input(shape = (5, 1))
x1 = LSTM(64)(input1)
x1 = Dense(10)(x1)
x1 = Dropout(rate = 0.2)(x1)

input2 = Input(shape = (5, 1))
x2 = LSTM(64)(input2)
x2 = Dense(5)(x2)
x2 = Dropout(rate = 0.2)(x2)

merge = Concatenate()([x1, x2])

output = Dense(1)(merge)

model = Model(inputs = [input1, input2], output = output)

model.summary()


# 3. 컴파일, 훈련 
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x_sam, x_hit], y_sam, epochs = 5)
hite = hite.fillna(method="bfill")#pandas에서는 행
print(samsung.shape)
print(hite.head())

# hite.iloc[0,1:5]=[10,20,30,40]
# hite.loc["2020-06-02","고가":"거래량"]=[20,30,40,50]

print(hite.head())
print(hite.describe())

hite=hite[:509]
samsung=samsung[:509]

samsung=samsung.sort_values(["일자"])
hite=hite.sort_values(["일자"])

def split1(datasets,timesteps):#samsung
    x_values=list()
    y_values=list()
    for i in range(len(datasets)-timesteps-1):#10-5-1
        x=datasets[i:i+timesteps]#0,1,2,3,4
        x=np.append(x,datasets[i+timesteps,0])#(509,26)
        y=datasets[i+timesteps+1]
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values),np.array(y_values)

# def split2(datasets,timesteps):#hite
#     x_values=list()
#     y_values=list()
#     for i in range(len(datasets)-timesteps-1):#10-5-1
#         x=datasets[i:i+timesteps]
#         y=datasets[i+timesteps,0]
#         x_values.append(x)
#         y_values.append(y)
#     return np.array(x_values),np.array(y_values)

# # 1) 모델구성 

#x,y값 나누기

x_s,y_s=split1(samsung_array,5)
x_h,y_h=split1(hite_array,5)

# print(f"x_h.shape:{x_h.shape}")
# print(f"x_s.shape:{x_s.shape}")

#scaler 위해서 reshape
#했으나, 어펜드 하면서 취소.
# x_s= x_s.reshape(-1,x_s.shape[1]*x_s.shape[2])
# x_h= x_h.reshape(-1,x_h.shape[1]*x_h.shape[2])

scaler1 = StandardScaler()#삼성
x_s=scaler1.fit_transform(x_s)

scaler2=StandardScaler()#하이트
x_h=scaler2.fit_transform(x_h)


#train_data, test_data 나눔
x_h_train,x_h_test,y_h_train,y_h_test,x_train,x_test,y_train,y_test=tts(x_h,y_h,x_s,y_s,train_size=0.8)


# print(f"x_h.shape:{x_h.shape}")
# print(f"x_s.shape:{x_s.shape}")


# print(f"y_h.shape:{y_h.shape}")
# print(f"y_s.shape:{y_s.shape}")

# # 2) 모델구성 

#2-1)하이트

input1 = Input(shape=(26,))
dense1 = Dense(2000,activation="relu")(input1)
dense1 = Dense(1,activation="relu")(dense1)

#2-2)삼성
input2 = Input(shape=(6,))
dense2 = Dense(2000,activation="relu")(input2)
dense2 = Dense(1,activation="relu")(dense2)


merge = Concatenate()([dense1,dense2])

# output1 = Dense(10,activation="relu")(merge)
# output1 = Dense(1,activation="relu")(output1)

output2 = Dense(2000,activation="relu")(merge)
output2 = Dense(1,activation="relu")(output2)

model = Model(inputs= [input1,input2], outputs = output2)

# model.summary()

# 1)나머지 4개의 값을 가져온다-samsung
# 2)바로 그 다음 값을 예측한다.

# # 3)트레이닝
# early= EarlyStopping(monitor="val_loss",patience=max(epochs//20,5))
model.compile(loss = "mse",optimizer="adam")
#하이트, 삼성 순서로 데이터 넣음
model.fit([x_h_train,x_train],y_train,batch_size=1,epochs=epochs,validation_split=0.2,verbose=0)#,callbacks=[early])
