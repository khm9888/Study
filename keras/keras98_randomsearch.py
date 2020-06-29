from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential,Model
from keras.layers import Dropout, MaxPool2D, Conv2D, Flatten, Dense, Input
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn. import scroe
#%%
#데이터 입력

# print(type(mnist.load_data()))

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)#(60000, 28, 28)
print(x_test.shape)#(10000, 28, 28)

# x_train= x_train.reshape(-1,x_train.shape[1],x_train.shape[2],1)/255
# x_test= x_test.reshape(-1,x_test.shape[1],x_test.shape[2],1)/255
x_train= x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])/255 #2차원
x_test= x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])/255 #2차원

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#모델 구성

def build_model(drop = 0.5, optimizer = "adam"):
    inputs = Input(shape=(28*28,),name="input")
    x = Dense(512,activation="relu", name="hidden1")(inputs)
    x = Dropout(drop)(x)
    x = Dense(256,activation="relu", name="hidden2")(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation="relu", name="hidden3")(x)
    x = Dropout(drop)(x)
    x = Dense(64,activation="relu", name="hidden4")(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation="softmax")(x)
    
    model = Model(inputs = inputs,outputs = outputs)
    model.compile(loss ="categorical_crossentropy",optimizer=optimizer,metrics=["acc"])
    # model.fit(x_train,y_train,batch_size=100)
    
    return model
    
def create_hyperparameter():
    # batches = list(range(10,60,10))#5번
    optimizers = ["rmsprop","adam","adadelta"]#3번
    # dropout = np.linspace(0.1,0.3, 2)#5번
    # return {'''"batch_size":batches,''' "optimizer":optimizers, "drop": dropout }
    return  {"optimizer":optimizers}
    
    
model = KerasClassifier(build_fn=build_model,verbose=1)
# build_model()

hyperparameters = create_hyperparameter()

search = RandomizedSearchCV(model,hyperparameters,cv=3)
search.fit(x_train,y_train)

print(search.best_params_)

     

# %%
