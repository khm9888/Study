from keras.models import Model
from keras.layers import Dense, Input
import numpy as np

#1.데이터 입력
x_data = [[0,0],[1,0],[0,1],[1,1]]
x_data = np.array(x_data)
y_data = [0,1,1,0]
y_data = np.array(y_data)



from keras.utils import np_utils

# y_data=np_utils.to_categorical(y_data)

#2.모델

# model =SVC()
input1= Input(shape=(2,))
dense1=Dense(1000,activation="relu")(input1)
dense1=Dense(1000,activation="relu")(dense1)
dense1=Dense(1,activation="sigmoid")(dense1)
model = Model(inputs = input1,outputs=dense1)

#3.훈련

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])
model.fit(x_data,y_data,epochs=50)

#4. 평가, 예측

loss,acc=model.evaluate(x_data,y_data)

x_test = [[0,0],[1,0],[0,1],[1,1]]
x_test = np.array(x_test)

y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict,axis=-1)
print(x_test,"의 예측결과",y_predict)
print("acc = ",acc)