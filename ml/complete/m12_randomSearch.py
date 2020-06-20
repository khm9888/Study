import pandas as pd
from sklearn.model_selection import train_test_split as tts, KFold, cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
# from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
x_test=x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])

paraemeters = {'bootstrap': [True, False]}



kfold = KFold(n_splits=5,shuffle=True)

model = RandomizedSearchCV(RandomForestClassifier(),paraemeters,cv=kfold,n_jobs=-1)

model.fit(x_train,y_train)

# y_pre = model.predict(x_test)

print(f"최적의 매개변수 = {model.best_estimator_}")

# print(f"최종 정답률 = {accuracy_score(y_test,y_pre)}")

