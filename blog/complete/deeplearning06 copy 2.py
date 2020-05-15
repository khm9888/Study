#데이터 저장

import numpy as np 

x1=np.array([range(100),range(200)])
x2=np.array([range(100),range(200)])
y=np.array([range(200),range(300)])

x1=x1.transpose()
x2=x2.transpose()
y=y.transpose()

from sklearn.model_selection import train_test_split as tts

x1_train,x1_test,y_train,y_test,x2_train,x2_test=tts(x1,y,x2,train_size=0.8)
#valid 함수는 fit함수에서 나눌 예정입니다.

#모델 구성

from keras.models import Model
from keras.layers import Dense,Input,Concatenate

#model1

#함수적 모델_1

input_1=Input(shape=(2,))
dense_1=Dense(5, activation="relu")(input_1)
dense_2=Dense(3)(dense_1)
dense_3=Dense(2)(dense_2)
output_1=Dense(2)(dense_3)

#model_1=Model(inputs=input_1,outputs=output_1)

#model_1.summary()

#함수적 모델 2

input_2=Input(shape=(2,))
dense_4=Dense(5, activation="relu")(input_2)
dense_5=Dense(3)(dense_4)
dense_6=Dense(2)(dense_5)
output_2=Dense(2)(dense_6)

#model_2=Model(inputs=input_2,outputs=output_2)

#model_2.summary()


# 모델 앙상블(concatenate)
con = Concatenate()([output_1,output_2])
#con = Concatenate()([model_1,model_2])
#실수한 부분, 완성된 모델을 넣었다.


model1=Dense(5)(con)
model2=Dense(3)(model1)
output_c=Dense(2)(model2)

model_c=Model(inputs=[input_1,input_2],outputs=output_c)

model_c.summary()