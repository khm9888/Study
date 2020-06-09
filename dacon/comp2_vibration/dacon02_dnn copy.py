import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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


print(train_features.shape)
print(train_features.shape)

x=train_features
y=train_target
x_pre=test_features

# print(train_features.shape)#(1050000, 5) #id,Time,S1,S2,S3,S4 #x_train,x_test
# print(train_target.shape)#(2800, 4) #id,X,Y,M,V id :1~2799 #y_train,y_test


# print(test_features.shape)#(262500, 5) #id,Time,S1,S2,S3,S4 #x_pre
# print(sample_submission.shape)#(700, 4) #id,X,Y,M,V id:2800~3499 #y_pre

# print("train_features.info",train_features.info())



#1) 데이터 전처리

x=x.values
y=train_target.values #(2800, 4)
x_pre=x_pre.values


x= x.reshape(-1,375,5)
x_pre= x_pre.reshape(-1,375,5)

x_train,x_test,y_train,y_test =tts(x,y,train_size=0.3) 

x_train  = x_train.reshape(x_train.shape[0]*x_train.shape[1]*5)
x_test  = x_test.reshape(x_test.shape[0],x_test.shape[1]*5)
x_pre  = x_pre.reshape(x_pre.shape[0],x_pre.shape[1]*5)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pre = scaler.transform(x_pre)


#모델구성
model = Sequential()

model.add(Dense(256,input_shape=(5,),activation="relu"))
model.add(Dense(128,activation="leaky "))
model.add(Dense(64,activation="leaky "))
model.add(Dense(32,activation="leaky "))
model.add(Dense(4,activation="leaky "))

model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=300,batch_size=32,validation_split=0.4)

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
submission.to_csv(f"./dacon\comp2_vibration\submission_{__file__[-14:-3]}.csv")