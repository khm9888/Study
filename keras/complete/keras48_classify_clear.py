import numpy as np

#회귀모델?


#데이터 구성

x = np.array(range(1,11))
print([1,0]*5)
y = np.array([1,0]*5)

#모델구성

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(64, input_shape=(1,)))
model.add(Dense(64,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

#model.summary()

#트레이닝

from keras.callbacks import EarlyStopping

early = EarlyStopping(monitor="mse", patience=10, mode="min")

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x,y,batch_size=1,epochs=100)

#테스트

mse,acc=model.evaluate(x,y,batch_size=1)
y_pre=model.predict(x)
y_pre=y_pre.reshape(10,)
print(f"y_pre(before):{y_pre}")
# for i,j in enumerate(y_pre):
#     if j>0.5:
#         y_pre[i]=1
#     else:
#         y_pre[i]=0

y_pre=[int(round(i)) for i in y_pre]
print(f"y_pre:{y_pre}")