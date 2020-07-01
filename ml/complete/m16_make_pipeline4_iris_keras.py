#iris를 케라스 파이프라인 구성
#당연히 Randomized SearchCV로 구성
#keras98참조할것

import numpy as np
from sklearn.datasets import load_iris
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
iris = load_iris()
x = iris.data
y = iris.target
print(x.shape)  #(150, 4)
print(y.shape)  #(150, )
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=43)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)/255
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)/255
# x_train = x_train.reshape(x_train.shape[0], 28*28)/255
# x_test = x_test.reshape(x_test.shape[0], 28*28)/255
print(x_train.shape) # (120, 4)
print(x_test.shape)  # (30, 4)
print(y_train.shape)  # (120, )
print(y_test.shape)  # (30,)



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)   
print(y_train.shape) #120 3
print(y_test.shape)  #30 3 

#2. 모델   (모델 자체를 함수로 만든다.) 

# Model 변수
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(4, ), name='input')
    x = Dense(512, activation='relu', name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name= 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, name= 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model

# Parameter 변수
def create_hyper_parameters():
    batches = [128, 256, 512]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.5, 5]
    epochs = [1, 10, 100]
    return{"kerasclassifier__batch_size" : batches, "kerasclassifier__optimizer" : optimizers, 
           "kerasclassifier__drop" : dropout, "kerasclassifier__epochs": epochs}
    # return{"model__batch_size" : batches, "model__optimizer" : optimizers, 
    #        "model__drop" : dropout, "model__epochs": epochs}
## pipe = Pipeline([('scaler',MinMaxScaler()), ('model',model)])'
# pipe라인을 붙일 땐 이렇게 하고

model = KerasClassifier(build_fn = build_model, verbose = 1)  # sklearn 에서 쓸 수 있게 랩핑을 함
parameter = create_hyper_parameters()


# pipe = Pipeline([('scaler',MinMaxScaler()), ('model',model)])
pipe = make_pipeline(MinMaxScaler(), model)
# dict_keys(['memory', 'steps', 'verbose', 'scaler', 'model', 'scaler__copy', 'scaler__feature_range', 'model__verbose', 'model__build_fn'])
# dict_keys(['memory', 'steps', 'verbose', 'minmaxscaler', 'kerasclassifier', 'minmaxscaler__copy', 'minmaxscaler__feature_range', 'kerasclassifier__verbose', 'kerasclassifier__build_fn'])

print(pipe.get_params().keys())

search = RandomizedSearchCV(pipe, parameter, cv=3)

search.fit(x_train, y_train)

acc = search.score(x_test, y_test)

print(search.best_params_)
print("최종 정답률 : ", acc)

