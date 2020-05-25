#1.데이터 저장

import numpy as np # numpy 모듈 을 가져와서 np 라고 지칭한다.

x_train = np.array([i for i in range(1,11)])
y_train = np.array([i for i in range(1,11)])

x_valid = np.array([i for i in range(21,31)])
y_valid = np.array([i for i in range(21,31)])

x_test = np.array([i for i in range(41,51)])
y_test = np.array([i for i in range(41,51)])

#입력값과 출력값 (x,y) 한 쌍이 존재하고 있는데
#딥러닝은 훈련해야하는 변수(여기서는 train)
#검증 테스트를 해야하는 변수(train)
#실질적 예상치를 얻어야 하는 실행값(test)

#이렇게 3개 이상의 값을 권장한다.

#2.모델구성 - 딥러닝을 하기 위해서는 학습 및 처리를 위한 모델을 만든다.layer및 node를 추가한다.

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

#kerea에 속한 두 모듈을 가져옵니다.(models,layers)
#하지만 위의 numpy와 형식이 다른 이유는 Sequential과 Dense라는 것을 numpy처럼 클래스나 함수 앞에 붙여주지 않아도 된다.
#ex> np.array <-> Sequential() - > 모듈이름을 붙여준다면? keras.models.Sequential() 이라고 써야합니다.

model.add(Dense(5,input_dim=1,activation="relu"))#입력값 1, 5개의 node
model.add(Dense(100))#3개의 node
model.add(Dense(1))#1개의 node 및 출력값

model.summary()
#만들어진 모델의 node 개수나 parameter의 개수 등을 보여줍니다.(터미널 창 통해서)

#3.훈련- 만들어진 모델을 학습을 시킨다.
6
model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=100,validation_data=(x_valid,y_valid))
#학습은 train이라는 변수로, 검증 테스트는 valid 변수로 값을 넣어준다. 
#epochs라는 매개변수-parameter-는 학습 반복횟수!

#4.평가 및 예측

loss,acc = model.evaluate(x_test,y_test)
#compile 함수에서 loss라고 되어있는 mse값이 loss 변수에, metirics에 들어간 값이 acc 변수에 들어간다.

print(f"loss : {loss}")
print(f"acc : {acc}")

output = model.predict(x_test)
#실제로 얻어야하는 값에 대한 입력값(x_test)를 넣어 예상치를 얻습니다.
print(f"output : \n {output}")


