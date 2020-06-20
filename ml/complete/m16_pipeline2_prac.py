import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.layers import Dense, Input
from sklearn.metrics import r2_score,mean_squared_error as mse,accuracy_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC,SVC
from keras.utils import np_utils
from sklearn.datasets import load_iris
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline,make_pipeline
from keras.models import Sequential,Model
from keras.layers import Dense,Input

datasets = load_iris()

# print(datasets.keys())

x= datasets.data
y= datasets.target


print(dir(Pipeline))


x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)


def build_model(drop=0.5,optimizer="adam"):
    i = Input(shape=(4,))
    d = Dense(1000,activation="relu")(i)
    d = Dense(1,activation="relu")(d)
    model = Model(inputs =i,outputs = d)
    model.compile(loss="mse",optimizer=optimizer,metrics=["acc"])
    return model

def create_hyper_parameter():
    batches = [1,2,3]
    
    return {"model__batch_size" : batches}
    # return dict(model__batch_size=batches)

model = KerasClassifier(build_model)
parameters = create_hyper_parameter()

pipe=Pipeline([("scaler",MinMaxScaler()),("model",model)])

search = RandomizedSearchCV(pipe,parameters,cv=3)

search.fit(x_train,y_train)

# loss,acc=search.evaluate(x_test,y_test)?
print(search.best_params_)