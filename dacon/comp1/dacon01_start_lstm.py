import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Dropout,LSTM
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

#0~1 데이터 전처리, 데이터 입력

    

train = pd.read_csv("./dacon/comp1/train.csv",index_col=0,header=0,encoding="cp949")
test = pd.read_csv("./dacon/comp1/test.csv",index_col=0,header=0,encoding="cp949")
submission = pd.read_csv("./dacon/comp1/sample_submission.csv",index_col=0,header=0,encoding="cp949")

# for i in range(len()):
    
    
print(train.shape)#(10000, 75) train,test
print(test.shape)#(10000, 71) x_predict
print(submission.shape)#(10000, 4) y_predict


print(train.isnull().sum())
train = train.interpolate()#보간법 // 선형보간

# print(dir(train))
print(train.isnull().sum())
test = test.interpolate() #보간법 // 선형보간

print(type(train))

train = train.fillna(0,axis=1)
test = test.fillna(0,axis=1)


x = train.values[:,:-4]
y = train.values[:,-4:]

x_train,x_test,y_train,y_test =tts(x,y,train_size=0.3) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train=x_train.reshape(-1,x_train.shape[1],1)
x_test=x_test.reshape(-1,x_test.shape[1],1)


#2.모델 구성
model = Sequential()

model.add(LSTM(256,input_shape=(71,1),activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4,activation="relu"))

# np.fillna

#3.훈련

# model = RandomizedSearchCV(model,{},cv=3)

model.compile(loss="mae",optimizer="adam")
model.fit(x_train,y_train,epochs=30,batch_size=32,validation_split=0.3)

#4.평가 및 예측
loss = model.evaluate(x_test,y_test,batch_size=10)
y_pre = model.predict(test)

a = np.arange(10000,20000)
submission = pd.DataFrame(y_pre,a)

# submission=pd.DataFrame(data=y_pre,index=submission.)

print(submission.head)

print(f"loss:{loss}")
# print(f"r2:{r2_score(y_test,y_pre)}")
# print(f"y_test[0:20]:{y_test[0:20]}")
# print(f"y_pre[0:20]:{y_pre[0:20]}")
submission.to_csv("dacon\comp1\sample_submission.csv")
#y_pre.to_csv

# submit = pd.DataFrame({})

