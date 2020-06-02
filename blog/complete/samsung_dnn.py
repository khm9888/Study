import numpy as np
import pandas as pd

samsung_data = pd.read_csv("./part10/samsung.csv",header=0,sep=",",index_col=0,encoding='cp949')#DataFrame
df1 = pd.read_csv("./part10/\kospi200.csv",index_col=0,header=0,encoding='cp949',sep=',')

print(samsung_data.shape)
print(df1.shape)

print(samsung_data.head())

samsung_data = samsung_data.sort_values(['일자'],ascending=[True])

print(samsung_data.head())


# print(samsung_data.keys())

# for i in range(len(samsung_data.index)):
#     for j in range(len(samsung_data.iloc[i])):
#         samsung_data.iloc[i,j]=int(samsung_data.iloc[i,j].replace(",",""))



samsung_data = samsung_data.sort_values(['일자'],ascending=[True])

# data=samsung_data.values#np.array

# np.save("./data/x_1.npy",arr=data)
# dataset=np.load("./data/x_1.npy",allow_pickle=True)

# print(dataset)

# print(type(dataset[0,4]))

# def split(dataset,time_steps,feature):
#     x_list=[]
#     y_list=[]
#     for i in range((len(dataset)-time_steps-feature+1)):#10-5-1=5
#         x=dataset[i:i+time_steps]
#         y=dataset[i+time_steps:i+time_steps+feature,0]
#         x_list.append(x)
#         y_list.append(y)
#     return np.array(x_list),np.array(y_list)



# #1)데이터 구성

# x,y=split(dataset,5,1)

# #디멘션확인
# print(x.shape)
# print(y.shape)

# #scaler를 적용하기 위해서 적용
# x=x.reshape(-1,x.shape[1]*x.shape[2])

# #스케일러 적용
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# from sklearn.model_selection import train_test_split as tts

# x_train,x_test,y_train,y_test=tts(x,y,shuffle=0,train_size=0.8)

# #2)모델구성

# from keras.models import Model
# from keras.layers import Dense,Input

# input1 = Input(shape=(25,))
# dense = Dense(1000,activation="relu")(input1)
# dense = Dense(1,activation="relu")(dense)

# model=Model(inputs=input1, output=dense)

# #3)트레이닝

# from keras.callbacks import EarlyStopping

# early=EarlyStopping(monitor="val_loss",patience=5)
# model.compile(loss="mse", optimizer="adam")
# model.fit(x_train,y_train,batch_size=1,epochs=150,validation_split=0.1,callbacks=[early])

# #4)테스트

# loss=model.evaluate(x_test,y_test,batch_size=1)
# y_pre = model.predict(x_test)

# from sklearn.metrics import r2_score,mean_squared_error as mse


# def rmse(y_test,y_pre):
#     return np.sqrt(mse(y_test,y_pre))

# y_test=y_test.reshape(-1,)
# y_pre=y_pre.reshape(-1,)

# print(__file__)
# print(f"loss:{loss}")
# print(f"y_test[0:10]:{y_test[0:10]}")
# print(f"y_pre[0:10]:{y_pre[0:10]}")

# print(f"mse:{mse(y_test,y_pre)}")
# print(f"rmse:{rmse(y_test,y_pre)}")
# print(f"r2_score:{r2_score(y_test,y_pre)}")


# '''
# y_test[0:10]:[56100 51000 43900 45250 44050 43850 56700 59500 46800 38500]
# y_pre[0:10]:[55780.61  51134.49  44933.492 45399.273 44241.434 43247.508 56668.207
#  60074.023 46319.695 38586.777]
# rmse:510.2521098003487
# r2_score:0.9869797347531619
# PS D:\Study>
# '''