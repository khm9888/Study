import pandas as pd
from sklearn.model_selection import train_test_split as tts, KFold, cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np
dataset=load_breast_cancer()
print(dataset.keys())
x=dataset.data
y=dataset.target

# print(type(x_train))#array

# iris = pd.read_csv("./data/csv/iris.csv",index_col=None,header=0)

# iris.info()
# print(iris.head())

# x=iris.iloc[:,0:4]
# y=iris.iloc[:,4]
# print(x)
# x= iris[]
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)

# x_train=x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
# x_test=x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])

paraemeters = [
    {'bootstrap': [True, False],
 'max_depth': [10, 30,  50,  70,   100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 600,  1600, 2000]}
]



print(np.sort.__code__.co_varnames)
input()
kfold = KFold(n_splits=5,shuffle=True)

print(kfold.__code__.co_varnames)
input()


model = GridSearchCV(RandomForestClassifier(),paraemeters,cv=kfold,n_jobs=-1)

model.fit(x_train,y_train)

y_pre = model.predict(x_test)

print(f"최적의 매개변수 = {model.best_estimator_}")
print(f"최종 정답률 = {accuracy_score(y_test,y_pre)}")

