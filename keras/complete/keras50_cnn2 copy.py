from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten

model = Sequential()

model.add(Conv2D(10,(2,2),input_shape=(10,10,1)))#(9,9,10)
model.add(Conv2D(7,(3,3)))#(7,7,7)
model.add(Conv2D(5,(2,2)))#(6,6,5)
# model.add(Conv2D(5,(2,2),strides=2,padding="same"))#(3,3,5) #padding 날라감

# model.summary()

model.add(MaxPooling2D(pool_size=2,strides=3))
# model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.summary()

