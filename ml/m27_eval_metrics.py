from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split as tts
import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

boston = load_boston()


x= boston.data
y= boston.target

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

xgb = XGBRegressor(n_estimators=10,learning_rate = 0.1)

xgb.fit(x_train,y_train,verbose=True, eval_metric=["rmse","logloss"],eval_set=[(x_train,y_train), (x_test, y_test)],early_stopping_rounds=20)
#rmse,mae,logloss,error,auc


y_pre = xgb.predict(x_test)

r2 = r2_score(y_test,y_pre)
score = xgb.score(x_test,y_test)
result = xgb.evals_result()
print(__file__)
print(result)
print("r2")
print(r2)
print("score")
print(score)
