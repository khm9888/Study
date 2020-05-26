import numpy as np



from sklearn.preprocessing import OneHotEncoder

#회귀모델?


#데이터 구성

x = np.array(range(1,11))
l=list(range(1,6))*2#(10,)
y = np.array(l)
# y = np.array(l).reshape(-1,1)

from keras.utils import np_utils

#option1
y=np_utils.to_categorical(y)
print("-"*30+"before"+"-"*30)
print(f"y.shape:{y.shape}")
print(f"y:{y}")



# enc= OneHotEncoder()
# enc.fit(y)
# y= enc.transform(y).toarray()#리스트를 array로 바꿔주는 함수

# print(f"y.shape:{y.shape}")
# print(f"y:{y}")


#모델구성

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

# def divide(val):
#     return val%5

model.add(Dense(64, input_shape=(1,)))
# model.add(Dense(64,activation="relu"))
# model.add(Dense(64,activation="relu"))
# model.add(Dense(64,activation="relu"))
model.add(Dense(6,activation="softmax"))#다중분류

#model.summary()

#트레이닝



from keras.callbacks import EarlyStopping

early = EarlyStopping(monitor="mse", patience=10, mode="min")

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x,y,batch_size=1,epochs=100)

#테스트

mse,acc=model.evaluate(x,y,batch_size=1)
y_pre=model.predict(x)
y_pre=np.argmax(y_pre,axis=1).reshape(-1,)

print()
print(f"y_pre:{y_pre}")




