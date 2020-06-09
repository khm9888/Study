import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Dropout,LSTM,Conv1D,Flatten
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
# print(train_features.groupby(['id']).max().head())
# print(train_features.groupby(['id']).max().shape)
# print(train_features.groupby(['id']).min().shape)

# print(train_features.groupby(['id']).max().iloc[:,1:].shape)
# print(train_features.groupby(['id']).min().iloc[:,1:].shape)

# x=pd.merge(train_features.groupby(['id']).max().iloc[:,1:],train_features.groupby(['id']).min().iloc[:,1:],on="id")
# x_pre=pd.merge(test_features.groupby(['id']).max().iloc[:,1:],test_features.groupby(['id']).min().iloc[:,1:],on="id")

# print("x.shape")
# print(x)



# print(train_features.shape)#(1050000, 5) #id,Time,S1,S2,S3,S4 #x_train,x_test
# print(train_target.shape)#(2800, 4) #id,X,Y,M,V id :1~2799 #y_train,y_test


# print(test_features.shape)#(262500, 5) #id,Time,S1,S2,S3,S4 #x_pre
# print(sample_submission.shape)#(700, 4) #id,X,Y,M,V id:2800~3499 #y_pre

# print("train_features.info",train_features.info())

# test_features.groupby(['id']).count()
# # 모든 id는 375개의 값을 가진다(375*700 = 262,500)
# train_features.groupby(['id','Time']).count()
# 모든 id는 동일한 시간 간격을 가지고 0~0.001496까지 기록되어있다.(4*(375-1) = 1496)


#1) 데이터 전처리

# x=x.values
x=train_features.values
y=train_target.values #(2800, 4)
x_pre=test_features.values


# x= x.reshape(-1,375,5)

x_train,x_test,y_train,y_test =tts(x,y,train_size=0.3) 

x_train  = x_train.reshape(x_train.shape[0]*x_train.shape[1],5)
x_test  = x_test.reshape(x_test.shape[0]*x_test.shape[1],5)
x_pre  = x_pre.reshape(x_pre.shape[0]*x_pre.shape[1],5)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pre = scaler.transform(x_pre)

x_train  = x_train.reshape(-1,x_train.shape[1],1)
x_test  = x_test.reshape(-1,x_test.shape[1],1)
x_pre  = x_pre.reshape(-1,x_pre.shape[1],1)

# DataFrame 형태
# # print(x.iloc[:,0])
# print(type(x))#np.array
# print(f"x.shape:{x.shape}") #x.shape:(2800, 375, 5)
# print(x_array[0])

'''
# for i in range(len(y.keys())):
#         fig=plt.figure()
        
#         ax4=fig.add_subplot(221)
#         txt=str(i)
#         ax4.set_title(txt)
#         ax4.hist(y_array[:,i])
        
#         ax1=fig.add_subplot(222)
#         txt=str(i)
#         ax1.set_title(txt)
#         ax1.hist(y_array[:,i])
        
#         ax2=fig.add_subplot(223)
#         txt=str(i)
#         ax2.set_title(txt)
#         ax2.hist(y_array[:,i])
        
#         ax3=fig.add_subplot(224)
#         txt=str(i)
#         ax3.set_title(txt)
#         ax3.hist(y_array[:,i])
        
#         plt.show()



# fig=plt.figure()

# ax4=fig.add_subplot(221)
# txt=str(0)
# ax4.set_title(txt)
# ax4.hist(y_array[:,0])

# ax1=fig.add_subplot(222)
# txt=str(1)
# ax1.set_title(txt)
# ax1.hist(y_array[:,1])

# ax2=fig.add_subplot(223)
# txt=str(2)
# ax2.set_title(txt)
# ax2.hist(y_array[:,2])

# ax3=fig.add_subplot(224)
# txt=str(3)
# ax3.set_title(txt)
# ax3.hist(y_array[:,3])

# plt.show()
'''
#모델구성
model = Sequential()

model.add(Conv1D(256,(2),input_shape=(8,1),activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(4,activation="relu"))

model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=200,batch_size=32,validation_split=0.4)

#4.평가 및 예측
loss = model.evaluate(x_test,y_test,batch_size=10)
y_pre = model.predict(x_pre)

a = np.arange(2800,3500)
submission = pd.DataFrame(y_pre,a)

# submission=pd.DataFrame(data=y_pre,index=submission.)

# print(submission.head)

print(f"loss:{loss}")
# print(f"r2:{r2_score(y_test,y_pre)}")
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")
submission.to_csv("dacon\comp1\sample_submission.csv")