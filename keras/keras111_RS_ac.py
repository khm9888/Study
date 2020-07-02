# 107번을 activation 넣어서 완성하시오

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test,y_test) = mnist.load_data()
# print(x_train.shape)    # (60000, 28, 28)
# print(x_test.shape)     # (10000, 28, 28)

#1-1. 데이터 전처리
x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])/255
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])/255
# /255 == MinmaxScaler와 같은 기능

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)    # (60000, 10)
# print(y_test.shape)     # (10000, 10)

from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam

#2-1. 모델 구성(함수)
# 랜덤서치를 사용할 것인데, 사이킷런에 있는 것이기 때문에 모델을 만들어야 함
# 랜덤서치의 첫번재 매개 변수가 model이기 때문에 모델 자체를 함수로 만들겠다
def build_model(drop=0.1, optimizer='Adam', learning_rate=0.1, activation='relu'):
    inputs = Input(shape=(28*28, ), name='input')
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation=activation, name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    opt = optimizer(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model
    # 모델 만든 함수에서는 compile까지만, fit은 그리드서치에서 실행(cv가 있기 때문에)
    # 사이킷런에 쓸수 있게 KerasClassifier를 wrapping 한 것(모델, 파라미터, cv을 wrap해서 사용하겠다)

# 랜덤서치에 들어갈 첫번째 모델 구성
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)
# build_fn: 호출가능한 함수 혹은 클레스 인스턴스

#2-2. 파라미터 구성(함수) 
# 랜덤서치의 두번째 매개변수인 parameter도 함수로 정의
def create_hyperparameters():
    batches = [100,200,300,400,500]
    optimizers = [RMSprop, Adam, Adadelta, SGD, Adagrad, Nadam]
    learning_rate = np.linspace(0.1,1.0,10).tolist()
    dropout = np.linspace(0.1,0.5,5).tolist()
    activation = ['tanh','relu','elu','softmax']
    # activation = [LeakyReLU()]
    return{"batch_size": batches, "optimizer":optimizers, "learning_rate": learning_rate, "drop": dropout, "activation": activation}
    # 하이퍼파라미터에 대한 매개변수들이 key, values라 딕셔너리형으로 return
hyperparameters = create_hyperparameters()

#2-3. 랜덤서치 만들기
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model, hyperparameters, cv=3) #n_jobs=-1)

search.fit(x_train, y_train)
print(search.best_params_)

acc = search.score(x_test, y_test)
print("acc: ", acc)