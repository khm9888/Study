#1.데이터 수집
import numpy as np

x1=np.array([range(1,101),range(101,201),range(201,301)])
x2=np.array([range(1,101),range(101,201),range(201,301)])
y1=np.array([range(1,101),range(101,201),range(201,301)])
y2=np.array([range(1,101),range(101,201),range(201,301)])

x1=np.transpose(x1)
x2=np.transpose(x2)
y1=np.transpose(y1)
y2=np.transpose(y2)

print(x1.shape)

from sklearn.model_selection import train_test_split as tts

x1_train,x1_test,y1_train,y1_test,x2_train,x2_test,y2_train,y2_test=tts(x1,y1,x2,y2,train_size=0.8)
# x_train,x_valid,y_train,y_valid=tts(x_train,y_train,train_size=0.75)
# 여기서 나누지 않고 <3단계 훈련> 과정에 나눌 예정입니다.

#2. 모델 구성

from keras.models import Model,Sequential
from keras.layers import Input,Dense,Concatenate,Add

input1 = Input(shape=(3,))#input
dense1= Dense(5, activation="relu")(input1)
dense2= Dense(4)(dense1)
output1= Dense(3)(dense2)#output

# model=Model(inputs=input1,outputs=output1)

input2 = Input(shape=(3,))#input
dense3= Dense(5, activation="relu")(input2)
dense4= Dense(4)(dense3)
output2= Dense(3)(dense4)#output

# model=Model(inputs=input2,outputs=output2)

# merge1=Concatenate()([output1,output2])
merge1=Add()([output1,output2])

model1=Dense(10)(merge1)
model2=Dense(5)(model1)
output1=Dense(2)(model2)

output1_1=Dense(3)(output1)
output1_1=Dense(5)(output1_1)
output1_1=Dense(3)(output1_1)

output1_2=Dense(3)(output1)
output1_2=Dense(5)(output1_2)
output1_2=Dense(5)(output1_2)
output1_2=Dense(3)(output1_2)

model=Model(inputs=[input1,input2],outputs=[output1_1,output1_2])

model.summary()

#3.훈련
model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
model.fit([x1_train,x2_train],[y1_train,y2_train],epochs=100,verbose=1)#,validation_split=0.25,verbose=3)

#데이터처리

mse1,mse2,mse3,acc1,acc2 = model.evaluate([x1_test,x2_test],[y1_test,y2_test])

print(f"mse1:{mse1},mse2:{mse2},mse3:{mse3}")
print(f"acc1:{acc1},acc2:{acc2}")

y1_predict,y2_predict=model.predict([x1_test,x2_test])

# print(f"y_predict:\n{y_predict}")
print(model.metrics_names)

from sklearn.metrics import mean_squared_error as mse, r2_score

def rmse(y_test,y_predict):
    return np.sqrt(mse(y_test,y_predict))

print(f"rmse:{rmse(y1_test,y1_predict)+rmse(y2_test,y2_predict)}")
print(f"r2:{r2_score(y1_test,y1_predict)+r2_score(y2_test,y2_predict)}")

