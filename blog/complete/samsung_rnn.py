import numpy as np
import pandas as pd

samsung_data = pd.read_csv("./part10/samsung.csv",header=0,sep=",",index_col=0,encoding='cp949')#DataFrame

print(samsung_data.keys())

for i in range(len(samsung_data.index)):
    for j in range(len(samsung_data.iloc[i])):
        samsung_data.iloc[i,j]=int(samsung_data.iloc[i,j].replace(",",""))


samsung_data = samsung_data.sort_values(['일자'],ascending=[True])

data=samsung_data.values#np.array

np.save("./data/x_1.npy",arr=data)
dataset=np.load("./data/x_1.npy",allow_pickle=True)

print(dataset)

print(type(dataset[0,4]))

def split(dataset,time_steps,feature):
    x_list=[]
    y_list=[]
    for i in range((len(dataset)-time_steps-feature+1)):#10-5-1=5
        x=dataset[i:i+time_steps]
        y=dataset[i+time_steps:i+time_steps+feature,0]
        x_list.append(x)
        y_list.append(y)
    return np.array(x_list),np.array(y_list)



#1)데이터 구성

x,y=split(dataset,5,1)

#디멘션확인
print(x.shape)
print(y.shape)

#scaler를 적용하기 위해서 적용
x=x.reshape(-1,x.shape[1]*x.shape[2])

#스케일러 적용
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

#3차원이기에 적용
x=x.reshape(-1,5,5)

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x,y,shuffle=0,train_size=0.8)

#2)모델구성

from keras.models import Model
from keras.layers import Dense,Input,LSTM

input1 = Input(shape=(5,5))
dense = LSTM(1000,activation="relu")(input1)
dense = Dense(1,activation="relu")(dense)

model=Model(inputs=input1, output=dense)

#3)트레이닝

model.compile(loss="mse", optimizer="adam")
model.fit(x_train,y_train,batch_size=1,epochs=100,validation_split=0.1)

#4)테스트

loss=model.evaluate(x_test,y_test,batch_size=1)
y_pre = model.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error as mse

def rmse(y_test,y_pre):
    return np.sqrt(mse(y_test,y_pre))

y_test=y_test.reshape(-1,)
y_pre=y_pre.reshape(-1,)

print(__file__)
print(f"loss:{loss}")
print(f"y_test[0:10]:{y_test[0:10]}")
print(f"y_pre[0:10]:{y_pre[0:10]}")

print(f"rmse:{rmse(y_test,y_pre)}")
print(f"r2_score:{r2_score(y_test,y_pre)}")


'''
d:\Study\samsung_rnn.py
loss:2804065.4715877757
y_test[0:10]:[46750 46400 49000 43400 50300 43750 45400 58800 40050 50600]
y_pre[0:10]:[46551.152 45648.484 48791.176 42251.918 52245.79  45442.61  47525.082
 58390.89  39115.035 53303.816]
rmse:1674.5330593119024
r2_score:0.8865737590900318
'''