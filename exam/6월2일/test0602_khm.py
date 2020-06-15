import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from keras.models import Model
from keras.layers import Dense, Input, Concatenate
from sklearn.metrics import r2_score,mean_squared_error as mse
from keras.callbacks import EarlyStopping

for epochs in range(10,121,5):
    hite_df=pd.read_csv("./data/csv/hite.csv",index_col=0,header=0,encoding="cp949",sep=",")
    samsung_df=pd.read_csv("./data/csv/samsung.csv",index_col=0,header=0,encoding="cp949",sep=",")

    hite_df=hite_df[:509]
    samsung_df=samsung_df[:509]

    # print(hite_df.tail())
    # print(samsung_df.tail())

    # print(hite_df.head())
    # print(samsung_df.head())


    hite_df = hite_df.sort_values(['일자'],ascending=[True])
    samsung_df = samsung_df.sort_values(['일자'],ascending=[True])
    
    # print(hite_df.tail())
    # print(samsung_df.tail())

    # print(f"hite_df.shape:{hite_df.shape}")
    # print(f"samsung_df.shape:{samsung_df.shape}")

    # cnt=0
    # for i in range(len(hite_df.index)):
    #     if type(hite_df.iloc[i,0])!=str:
    #         cnt+=1
    # print(cnt)
        

    for i in range(len(hite_df.index)):
        for j in range(len(hite_df.iloc[i])):
            if type(hite_df.iloc[i,j])==str:
                hite_df.iloc[i,j] = int(hite_df.iloc[i,j].replace(",",""))
        # else:
        #     print(i,type(hite_df.iloc[i,0]))


    for i in range(len(samsung_df.index)):
        if type(samsung_df.iloc[i,0])==str:
            samsung_df.iloc[i,0] = int(samsung_df.iloc[i,0].replace(",",""))
    # for i in range(5):
        
    #     print(type(hite_df.iloc[0,i]))
    # print(type(samsung_df.iloc[0,0]))


    # print(hite_df.head())
    # print(samsung_df.head())

    hite_array=hite_df.values
    samsung_array=samsung_df.values
    
    np.save("./hite_array.npy",arr=hite_array)
    np.save("./samsung_array.npy",arr=samsung_array)

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

    model.save(F"./{__file__[-15:-3]}-{epochs}.h5")
    # # 4)테스트
    D:\Study\exam\6월2일\test0602_khm.py

    loss = model.evaluate([x_h_test,x_test],y_test,batch_size=1)
    y_pre=model.predict([x_h_test,x_test])

    r2=r2_score(y_test,y_pre)
    
    if r2<0.5 or r2>1:
        pass
    else:
        y_pre=y_pre.reshape(-1,)
        y_test=y_test.reshape(-1,)
        print("-"*20,"start","-"*20)
        print(__file__)
        print(f"epochs:{epochs}")
        print(f"loss:{loss}")
        print(f"rmse:{np.sqrt(mse(y_test,y_pre))}")
        print(f"r2:{r2_score(y_test,y_pre)}")
        print(f"y_test[:10]:{y_test[:10]}")
        print(f"y_pre[:10]:{y_pre[:10]}")
        
        
        hite=hite_array[-6:-1]
        hite=hite.reshape(25,)
        hite=np.append(hite,hite_array[-1,0])
        
        samsung=samsung_array[-6:-1]
        samsung=samsung.reshape(5,)
        samsung=np.append(samsung,samsung_array[-1,0])

        # print(hite.shape)
        # print(samsung.shape)
        
        hite = hite.reshape(1,26)#(26,1)
        samsung = samsung.reshape(1,6)
        
        samsung=scaler1.transform(samsung)
        hite=scaler2.transform(hite)
        
        # print(f"hite:{hite}")
        # print(f"samsung:{samsung}")
        
        y_truely_pre=model.predict([hite,samsung])
        
        print(f"예측값 : {y_truely_pre}")
        
        print("-"*20,"end","-"*20)
        print()
        print()

# -------------------- start --------------------
# c:\Users\bitcamp\Desktop\test0602_khm.py
# epochs:150
# loss:732544.2701677378
# rmse:855.8880027073651
# r2:0.9654045036060124
# y_test[:10]:[44500 46400 44300 51800 47050 38400 44800 44850 45150 49700]
# y_pre[:10]:[44011.02  46981.375 44925.44  50994.19  46415.508 38331.12  44746.96
#  44867.207 43993.87  49626.52 ]
# 예측값 : [[51477.04]]
# -------------------- end --------------------