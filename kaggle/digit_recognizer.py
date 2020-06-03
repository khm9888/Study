import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.datasets import mnist


# batch=100

# (x_train,y_train),,y_test) = mnist.load_data()

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

print(train.shape)
# print(test.head())

y_train=train["label"]
x_train=train.drop(labels = ["label"],axis = 1)#열?

del train

# print(f"type(x_train[0]):{type(x_train[0])}")
# print(f"x.shape:{x_train.shape}")

# print(f"type(x_train):{type(x_train)}")



x_train= x_train /255
test= test /255

x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)



from keras.utils import np_utils
# enc = OneHotEncoder()

y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

print(y_train.shape)
# y_train = y_train.reshape(-1,28,28,1)
# y_test = y_test.reshape(-1,28,28,1)

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x_train,y_train,train_size=0.9)

# y_train= y_train /255


# y_test= y_test /255
# print(f"x_train:{x_train}")


#모델구성
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout

model= Sequential()

# from keras.callbacks import EarlyStopping



model.add(Conv2D(40,(2,2),input_shape=(28,28,1),activation="relu",padding="same"))
model.add(Conv2D(40,(2,2),activation="relu",padding="same"))
# model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(40,(2,2),activation="relu",padding="same"))
model.add(Conv2D(40,(2,2),activation="relu",padding="same"))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(40,(2,2),activation="relu",padding="same"))
# model.add(Conv2D(40,(2,2),activation="relu",padding="same"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(10,activation="softmax"))

model.summary()

#훈련


print("-"*20+str("start")+"-"*20)
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["acc"])
model.fit(x_train,y_train,epochs=30,batch_size=100,validation_split=0.3)

#테스트

loss,acc = model.evaluate(x_test,y_test,batch_size=100)
y_pre=model.predict(test)

# y_test=np.argmax(y_test[0:10],axis=1)
# y_pre=np.argmax(y_pre[0:10],axis=1)


import pandas as pd

# y_pre = model.predict(test)
y_pre = np.argmax(y_pre, axis = 1)
y_pre = pd.Series(y_pre, name = 'Label')

submit = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), y_pre], axis = 1)
submit.to_csv("mySubmission_mnist_cnn.csv", index = False)

print(f"loss:{loss}")
print(f"acc:{acc}")
y_test=np.argmax(y_test, axis=1)
y_pre=y_pre.values

print(f"y_test[0:10]:{y_test[0:10]}")
print(f"y_pre[0:10]:{y_pre[0:10]}")
