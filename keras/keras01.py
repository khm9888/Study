import numpy as np

values=[]
for i in range(1,11):
    values.append(i)
x=np.array(values)
y=np.array(values)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(1,input_dim=1,activation='relu'))

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,epochs=500,batch_size=1)

loss,acc=model.evaluate(x,y,batch_size=1)


print(f'loss : {loss}')
print(f'acc : {acc}')