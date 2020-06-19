# 실습 1. train, test 분리할것. (8:2)
# 실습 2. 마지막 6개의 행을 predict로 만들고 싶다.
# 실습 3. validation을 넣을 것(train의 20%)


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten

#1. 데이터
a = np.array(range(1, 101)) 
size = 5                           # time_steps = 4
# print("a.shape :", a.shape)


def split_x(seq, size):    
    aaa = []
    for i in range(len(seq) - size + 1): 
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])   
    print(type(aaa))
    return np.array(aaa)   


dataset = split_x(a, size)
print("============================")
print(dataset)
print(dataset.shape)   #(96, 5) 
print(type(dataset))   

x = dataset[:90, 0:4]    # : 은 all [모든행이 들어가겠다, (0,1,2,3)] 6, 5 짜리 똔똔해서 모든행을 가져오고 0부터 3(4-1) 까지를 가져오겠다.
y = dataset[:90, 4]
x_predict = dataset[90:, 0:4]
print(x)  #(90,4)
print(y)  #(90,)
print(x_predict)


x = np.reshape(x, (90, 4, 1))  
print(x.shape)

x_predict = x_predict.reshape(6, 4, 1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=False, train_size=0.8)



print("------------train---------")
print(x.shape)
print("----------test-----------")
print(y.shape)
print("---------predict------------")
# print(x_predict.shape)




#2. 모델구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(4, 1)))     
model.add(Conv1D(10, 2, padding='same', activation='relu', input_shape=(4, 1)))    
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(333))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.summary()

'''
#3. 컴파일 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto') 

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x_train, y_train, epochs=14, batch_size=1, verbose=1) #, callbacks=[early_stopping])




#4. 평가, 예측
loss, acc = model.evaluate(x_train,y_train)

y_predict = model.predict(x_predict)

print('loss : ', loss)
print('mse :', acc)
print('y_predict :', y_predict)
'''
'''
loss :  0.0705373270644082
mse : 0.07053732872009277
y_predict : [[94.38281 ]
 [95.37513 ]
 [96.36747 ]
 [97.35979 ]
 [98.352104]
 [99.34442 ]]
'''