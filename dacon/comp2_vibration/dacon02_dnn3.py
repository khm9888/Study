import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Dropout,LSTM
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV


train_features = pd.read_csv("dacon\comp2_vibration\data\\train_features.csv",index_col=0,header=0,encoding="cp949")
train_target = pd.read_csv("dacon\comp2_vibration\data\\train_target.csv",index_col=0,header=0,encoding="cp949")
test_features = pd.read_csv("dacon\comp2_vibration\data\\test_features.csv",index_col=0,header=0,encoding="cp949")
sample_submission = pd.read_csv("dacon\comp2_vibration\data\\sample_submission.csv",index_col=0,header=0,encoding="cp949")


# print("train_features.groupby(['id']).max()")
# print(train_features.groupby(["id"]).max())
# print("train_features.groupby(['id']).min()")
# print(train_features.groupby(["id"]).min())
# print(train_features.groupby(['id']).max().shape)
# print(type(train_features.groupby(['id']).max()))
print(train_features.groupby(['id']).max().head())
print(train_features.groupby(['id']).max().shape)
print(train_features.groupby(['id']).min().shape)

print(train_features.groupby(['id']).max().iloc[:,1:].shape)
print(train_features.groupby(['id']).min().iloc[:,1:].shape)

# x=pd.merge(train_features.groupby(['id']).max().iloc[:,1:], train_features.groupby(['id']).min().iloc[:,1:],  on="id")
# x_pre=pd.merge(test_features.groupby(['id']).max().iloc[:,1:], test_features.groupby(['id']).min().iloc[:,1:], on="id")

# x=pd.merge(x, train_features.groupby(['id']).mean().iloc[:,1:], on="id")
# x_pre=pd.merge(x_pre, test_features.groupby(['id']).mean().iloc[:,1:], on="id")

x=train_features.groupby(['id']).mean().iloc[:,1:]
x_pre=test_features.groupby(['id']).mean().iloc[:,1:]

print("x.shape")
print(x)




# print(train_features.shape)#(1050000, 5) #id,Tie,S1,S2,S3,S4 #x_train,x_test
# print(train_target.shape)#(2800, 4) #id,X,Y,M,V id :1~2799 #y_train,y_test


# print(test_features.shape)#(262500, 5) #id,Time,S1,S2,S3,S4 #x_pre
# print(sample_submission.shape)#(700, 4) #id,X,Y,M,V id:2800~3499 #y_pre

# print("train_features.info",train_features.info())

# test_features.groupby(['id']).count()
# # 모든 id는 375개의 값을 가진다(375*700 = 262,500)
# train_features.groupby(['id','Time']).count()
# 모든 id는 동일한 시간 간격을 가지고 0~0.001496까지 기록되어있다.(4*(375-1) = 1496)


#1) 데이터 전처리

x=x.values
y=train_target.values #(2800, 4)
x_pre=x_pre.values


# x= x.reshape(-1,375,5)

x_train,x_test,y_train,y_test =tts(x,y,train_size=0.3) 

# x_train  = x_train.reshape(x_train.shape[0]*x_train.shape[1],5)
# x_test  = x_test.reshape(x_test.shape[0]*x_test.shape[1],5)

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pre = scaler.transform(x_pre)



#모델구성
model = Sequential()

model.add(Dense(256,input_shape=(4,),activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(4,activation="relu"))

model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=300,batch_size=32,validation_split=0.5)

#4.평가 및 예측
loss = model.evaluate(x_test,y_test,batch_size=10)
y_pre = model.predict(x_pre)
sample_submission.keys()
a = np.arange(2800,3500)
# a = pd.Series(a,index="id")
c=list(sample_submission.columns)
submission = pd.DataFrame(y_pre,a ,columns=sample_submission.columns)

# submission=pd.DataFrame(data=y_pre,index=submission.)

# print(submission.head)

print(f"loss:{loss}")
# print(f"r2:{r2_score(y_test,y_pre)}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")
submission.to_csv(f"./dacon\comp2_vibration\submission_{__file__[-14:-3]}_2.csv")