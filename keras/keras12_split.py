import numpy as np

x=np.array(range(1,101))
y=np.array(range(1,101))
#x,y의 범위만 잡아줍니다.

from sklearn.model_selection import train_test_split
#사이킷런(sklearn)에서 model_selection api 선택하고, train_test_split 함수를 가져온다/
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.6, shuffle=False)
#train 변수와 test 변수를 6:4로 나눈다.

x_test,x_valid,y_test,y_valid=train_test_split(x_test,y_test,train_size=0.5,shuffle=False)
#나누어진 test 함수를, test 5, valid 5로 1대1 비율로 나눈다.

