from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split as tts
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score,accuracy_score

boston = load_breast_cancer()


x= boston.data
y= boston.target

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

xgb = XGBClassifier(n_estimators=1000,learning_rate = 0.1)

xgb.fit(x_train,y_train,verbose=True, eval_metric="rmse",eval_set=[(x_train,y_train), (x_test, y_test)],early_stopping_rounds=20)
#rmse,mae,logloss,error,auc


y_pre = xgb.predict(x_test)

acc = accuracy_score(y_test,y_pre)
score = xgb.score(x_test,y_test)
result = xgb.evals_result()
print(__file__)
print(result)
print("acc")
print(acc)
print("score")
print(score)

import pickle
pickle.dump(xgb,open("./model/xgb_save/cancer.plckle.dat","wb"))

print(123)

model2 = pickle.load(open("./model/xgb_save/cancer.plckle.dat","rb"))

print(123)

acc = accuracy_score(y_test,y_pre)
print(acc)