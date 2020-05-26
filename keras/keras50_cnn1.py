from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()

model.add(Conv2D(10,(2,2),padding="same",input_shape=(10,10,1)))#filter,
model.add(Conv2D(7,(2,2),padding="same"))
model.add(Conv2D(5,(2,2),padding="same"))

model.summary()

# model2 = Sequential()

# model2.add(Conv2D(10,(2,2),padding="valid",input_shape=(10,10,1)))
# model2.add(Conv2D(7,(2,2),padding="valid"))
# model2.add(Conv2D(5,(2,2),padding="valid"))

# model2.summary()

