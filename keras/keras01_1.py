import numpy as np

x=np.array([i for i in range(1,11)])
y=np.array([i for i in range(1,11)])

# print(x)
# print(y)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(1,input_dim=1,activation='relu'))

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,epochs=500,batch_size=1)

loss,acc=model.evaluate(x,y,batch_size=1)

print(f"loss : {loss}")
print(f"acc : {acc}")

