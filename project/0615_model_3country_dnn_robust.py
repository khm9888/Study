import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense,Input
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split as tts
import seaborn as sns


price_df=pd.read_csv("project\pro1/test.csv",index_col=0,header=0)

# print(type(price_df))

# print("price_df")
# print(price_df)

# print("price_df.describe()")
# print(price_df.describe())

# print("price_df.info()")
# price_df.info()

# x_df=price_df.loc[price_df.index[1]:,["s&p 500","nikkei 225","shanghai"]]#가장 최근값 없앰
# y_df=price_df.loc[:price_df.index[-2],["kospi200"]]#가장 옛날값 없앰


price_array = price_df.values

def split(datasets,timesteps):
    x_values=list()
    y_values=list()
    for i in range(len(datasets)-timesteps):#10-6
        x=datasets[i:i+timesteps,1:]#0
        y=datasets[i+timesteps,0]
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values),np.array(y_values)

x,y=split(price_array,6)
x,x_pre,y,y_predict = x[:-1],x[-1],y[:-1],y[-1]
# print("x.shape")
x = x.reshape(-1,x.shape[1]*x.shape[2])
# print(x.shape)
# print("y.shape")
# print(y.shape)

x_pre = x_pre.reshape(-1,x_pre.shape[0]*x_pre.shape[1])

# print(x_pre)
# print(x_pre.shape)

x_train,x_test,y_train,y_test = tts(x,y,train_size=0.7)

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
x_pre=scaler.transform(x_pre)

# print(x_pre)

#모델

input1=Input(shape=(x.shape[1],))
dense=Dense(3000,activation="relu")(input1)
dense=Dense(1,activation="relu")(dense)#회귀

model = Model(inputs=input1,outputs=dense)

model.summary()

#트레이닝

model.compile(loss="mse", optimizer="adam")
model.fit(x_train,y_train,batch_size=10,epochs=300,validation_split=0.3,verbose=2)

#테스트

loss= model.evaluate(x_test,y_test,batch_size=100)

y_pre=model.predict(x_test)
y_pre = y_pre.reshape(-1,)

print(__file__)
print(f"loss:{loss}")
print(f"r2:{r2_score(y_test,y_pre)}")
# print(f"x_test.shape:{x_test.shape}")
# print(f"y_pre.shape:{y_pre.shape}")

print(f"y_test:{y_test[0:20]}")
print(f"y_pre:{y_pre[0:20]}")

y_values = pd.DataFrame(dict(y_test=y_test[0:20],y_pre=y_pre[0:20]),index=range(1,21))

# print(y_values)

plt.figure(figsize=(6,6))

sns.lineplot(x=y_values.index,y="y_test", data=y_values , label ="y_test" )
sns.lineplot(x=y_values.index,y="y_pre", data=y_values ,label="y_pre")

plt.xlabel("count")
plt.ylabel("y=kospi")


plt.title("y_test & y_pre")
plt.legend(loc=0)
plt.show()

y_pre_tomorrow = model.predict(x_pre)
y_pre_tomorrow = y_pre_tomorrow.reshape(-1,)

print("tomorrow`s kospi 200 :",y_pre_tomorrow)
print("tomorrow`s truly kospi 200 :",y_predict)
'''
loss:27.008601412747076
r2:0.9588118067808987
y_test:[232.45 273.88 269.79 300.07 335.49 318.51 253.37 276.61 279.87 265.35
 325.79 301.53 297.19 256.28 293.77 309.53 291.63 280.32 295.9  292.9 ]
y_pre:[224.97675 275.378   268.9614  296.59912 331.90863 317.7645  263.43417
 275.82385 278.84818 262.67288 322.40494 296.18347 297.0006  257.81317
 297.57944 304.67685 293.4476  274.75726 298.72748 297.3773 ]
tomorrow`s kospi 200 : [286.17407]
tomorrow`s truly kospi 200 : 280.46
'''
'''loss:22.820804491720565
r2:0.9627051191779331
y_test:[325.67 292.86 268.75 335.44 327.35 327.01 255.02 289.   293.7  266.65
 316.71 265.55 255.85 276.56 320.12 269.93 263.74 312.86 267.4  298.8 ]
y_pre:[330.2439  290.99133 271.85178 334.41452 328.15048 333.49796 248.9524
 286.93222 290.191   269.79324 320.0214  266.75583 259.2432  280.54288
 319.39877 271.031   264.7982  318.62424 272.56058 295.34888]
tomorrow`s kospi 200 : [285.29263]
tomorrow`s truly kospi 200 : 280.46'''
'''loss:20.023361914796254
r2:0.9676992168119598
y_test:[274.34 326.6  294.4  249.4  262.95 337.8  338.05 279.94 268.32 295.24
 288.43 264.44 289.85 296.95 268.16 296.19 327.61 324.52 313.94 259.62]
y_pre:[274.85233 322.46036 291.3811  255.69598 271.44635 335.14426 336.6804
 281.965   270.15814 299.25055 292.1756  262.83014 292.93848 300.19287
 265.04956 294.17758 331.9265  324.69424 308.64795 259.1942 ]
tomorrow`s kospi 200 : [285.58734]
tomorrow`s truly kospi 200 : 280.46'''


