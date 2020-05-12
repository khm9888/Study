from keras.models import Sequential
from keras.layers import Dense

import numpy as np

small=[]
big=[]

for i in range(1,11):
    small.append(i)
    big.append(i+100)
    
print(small)
print(small)
print(big)
print(big)


# x_train=np.array([1,2,3,4,5,6,7,8,9,10])
# y_train=np.array([1,2,3,4,5,6,7,8,9,10])

# x_test=np.array([101,102,103,104,105,106,107,108,109,110])
# y_test=np.array([101,102,103,104,105,106,107,108,109,110])

x_train=np.array(small)
y_train=np.array(small)

x_test=np.array(big)
y_test=np.array(big)

model=Sequential()

model.add(Dense(5,input_dim=1,activation='relu'))
model.add(Dense(3))
model.add(Dense(1,activation='relu'))


model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,validation_data=(x_train,y_train))
model.fit

loss,acc=model.evaluate(x_test,y_test,batch_size=1)

print(f"loss : {loss}")
print(f"acc : {acc}")

output=model.predict(x_test)
print(f'결과물 : {output}')