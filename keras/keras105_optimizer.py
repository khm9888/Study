import numpy as np

#데이터 입력
x = np.arange(1,5,1)
y = np.arange(1,5,1)

#모델구성

from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(1000,activation="relu",input_shape=[1,]))
model.add(Dense(1,activation="relu"))

#훈련
from keras.optimizers import adam,adadelta,nadam,rmsprop,sgd,adagrad
opt_list = [adam,adadelta,nadam,rmsprop,sgd,adagrad]


# optimizer = adam(0.001)
for i,o in enumerate(opt_list):
    optimizer = opt_list[i]()
    model.compile(loss="mse", optimizer=optimizer,metrics=["mse"])
    model.fit(x,y,epochs=10,verbose=0)

#평가
    loss,metric = model.evaluate(x,y)
    y_pre = model.predict(x)

    print(o)
    print("loss")
    print(loss)
    print("y_pre")
    print(y_pre)

'''
<class 'keras.optimizers.Adam'>
loss
1.9830808639526367
y_pre
[[0.61565006]
 [1.0387453 ]
 [1.4602594 ]
 [1.8810881 ]]
4/4 [==============================] - 0s 3ms/step
<class 'keras.optimizers.Adadelta'>
loss
0.06889418512582779
y_pre
[[1.1723261]
 [1.972656 ]
 [2.7678294]
 [3.562702 ]]
4/4 [==============================] - 0s 3ms/step
<class 'keras.optimizers.Nadam'>
loss
0.010266028344631195
y_pre
 [2.9945698]
 [3.9088004]]
4/4 [==============================] - 0s 3ms/step
<class 'keras.optimizers.RMSprop'>
loss
0.0017413946334272623
y_pre
[[1.0659176]
 [2.0318146]
 [2.9961953]
 [3.9600775]]
4/4 [==============================] - 0s 3ms/step
<class 'keras.optimizers.SGD'>
loss
0.0015122053446248174
y_pre
[[1.0623566]
 [2.031222 ]
 [2.9986997]
 [3.9655912]]
4/4 [==============================] - 0s 3ms/step
<class 'keras.optimizers.Adagrad'>
loss
0.0068343328312039375
y_pre
[[0.86707205]
 [1.9357145 ]
 [3.004974  ]
 [4.07423   ]]
 '''

