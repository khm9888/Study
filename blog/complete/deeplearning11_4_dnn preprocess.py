#part11

#kospi200 데이터를 이용한 삼성전자 주가예측

#삼성전자 주식 가격과 kospi 가격을 이용해서, 내일의 삼성전자 주가 예측

#삼성전자 데이터를 가지고 DNN / RNN 모델을 만들고 비교

#그 후 입력 데이터 둘(삼성전자, kospi200)을 DNN과 :LSTM 로 구성하고, 앙상블 다:1로 비교

#p294

#DNN

import numpy as np
import pandas as pd

df1 = pd.read_csv(".\part10\kospi200.csv",index_col=0,header=0,encoding='cp949',sep=',')

print(df1)
print(df1.shape)

df2 = pd.read_csv(".\part10\samsung.csv",index_col=0,header=0,encoding="cp949",sep=',')

print(df2)
print(df2.shape)

print(len(df1.index))

#거래량 str->int
for i in range(len(df1.index)):#총 주소 반환, len = 426
    df1.iloc[i,4]=int(df1.iloc[i,4].replace(',',''))
    
#모든 항목 str->int
for i in range(len(df2.index)):
    for j in range(len(df2.iloc[i])):
        df2.iloc[i,j]=int(df2.iloc[i,j].replace(',',''))
        
        
df1 = df1.sort_values(['일자'],ascending=[True])#날짜 순으로 sorting
df2 = df2.sort_values(['일자'],ascending=[True])

print(df1)
print(df2)

#pandas 데이터를 numpy로 변경

df1=df1.values #values를 사용해서 numpy로 바꿉니다.
df2=df2.values

print(type(df1),type(df2))
print(df1.shape,df2.shape)

np.save(".\part10\kospi200.npy",arr=df1)
np.save(".\part10\samsung.npy",arr=df2)

# np.save("/kospi200/data/kospi200.npy",arr=df1)
# np.save("/kospi200/data/samsung.npy",arr=df2)


#x값은 5개, y값은 1개.5일분씩 잘라서 할 예정
def split(dataset,time_steps,y_column):
    x_list=list()
    y_list=list()
    for i in range(len(dataset)-time_steps+1-y_column):
        x=dataset[i:i+time_steps]
        y=dataset[i+time_steps:i+time_steps+y_column,-2]
        x_list.append(x)
        y_list.append(y)
    return np.array(x_list),np.array(y_list)

x,y=split(df2,5,1)
x2,y2=split(df1,5,1)

print(f"{x[0]} \n {y[0]}")
print(f"x.shape:{x.shape}")
print(f"y.shape:{y.shape}")

x=x.reshape(x.shape[0],x.shape[1]*x.shape[2])

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x,y,train_size=0.7)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


print(f"type(x_train_scaled):{type(x_train_scaled)}")
print(x_train_scaled[0, :])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#모델 생성
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(64,input_shape=(25,),activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(1))

#트레이닝

from keras.callbacks import EarlyStopping

early=EarlyStopping(patience=20)

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(x_train_scaled,y_train,epochs=100,batch_size=1,validation_split=0.1,callbacks=[early])

#테스트

mse,acc=model.evaluate(x_test_scaled,y_test,batch_size=1)

print(f"mse:{mse}")
print(f"acc:{acc}")

y_pre=model.predict(x_test_scaled)

print(f"y_pe.len:{len(y_pre)}")

for i in range(5):
    print(f"종가 : {y_test[i]} / 예측가 : {y_pre[i]}")


