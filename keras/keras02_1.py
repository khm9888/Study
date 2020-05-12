import numpy as np

x_train=np.array([i for i in range(1,11)])
y_train=np.array([i for i in range(1,11)])
x_test=np.array([i for i in range(101,111)])
y_test=np.array([i for i in range(101,111)])

# print(x)
# print(y)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(5,input_dim=1,activation='relu'))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
# model.compile(loss='mse',optimizer='adam',metrics=['mse'])

model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_train,y_train))

loss,acc=model.evaluate(x_test,y_test,batch_size=1)

print(f"loss : {loss}")
print(f"acc : {acc}")

output=model.predict(x_test)
print(f"output: \n{output}")