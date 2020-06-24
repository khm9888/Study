from sklearn.model_selection import train_test_split as tts
import numpy as np
from sklearn.datasets import load_iris

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
# 1.회귀
# 2.이진분류
# 3.다중분류

# eval 에 "loss"와 다른 지표 1개 더 추가
# earlystopping 적용
# plot 으로 그릴 것

# 4. 결과는 주석 하단

dataset = load_iris()

x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

lgbm = LGBMClassifier(n_estimators=100,learning_rate = 0.1,n_jobs=-1,objective='multi:multiclass')

# print(lgbm.get_params())

lgbm.fit(x_train,y_train,verbose=False, eval_metric=["multi_logloss","multi_error"],eval_set=[(x_train, y_train), (x_test,y_test)],early_stopping_rounds=20)


#rmse,mae,logloss,error,auc


y_pre = lgbm.predict(x_test)

r2 = r2_score(y_test,y_pre)
score = lgbm.score(x_test,y_test)
print(__file__)
print("r2")
print(r2)
print("score")
print(score)


#6)selectFromModel

thresholds = np.sort(lgbm.feature_importances_)

idx_max = -1
max = r2

for idx,thresh in enumerate(thresholds):
    #데이터 전처리
    selection = SelectFromModel(lgbm,threshold=thresh,prefit=True)
    #1)데이터입력
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    #2)모델구성
    selection_model = LGBMClassifier(n_estimators=100,learning_rate = 0.1,n_jobs=-1,objective='multi:multiclass')

    
    #3)훈련
    selection_model.fit(selection_x_train,y_train,verbose=False, eval_metric=["multi_logloss","multi_error"],eval_set=[(selection_x_train, y_train), (selection_x_test,y_test)],early_stopping_rounds=20)
    
    #4)평가 및 예측
    y_pre = selection_model.predict(selection_x_test)
    r2 = r2_score(y_test,y_pre)
    print("idx")
    print(idx)
    print("r2")
    print(r2)
    if max<=r2:
        max=r2
        idx_max=idx
print(__file__)
print("idx_max")
print(idx_max)
print("max")
print(max)

selection = SelectFromModel(lgbm,threshold=thresholds[idx_max],prefit=True)
selection_x_train = selection.transform(x_train)
selection_x_test = selection.transform(x_test)
#2)모델구성
selection_model = LGBMClassifier(n_estimators=100,learning_rate = 0.1,n_jobs=-1,objective='multi:multiclass')

import pickle
import joblib

#3)훈련
selection_model.fit(selection_x_train,y_train,verbose=False, eval_metric=["multi_logloss","multi_error"],eval_set=[(selection_x_train, y_train), (selection_x_test,y_test)],early_stopping_rounds=20)
path=f"./model/lgbm_save/{__file__[-(3+15):-3]}-idx{idx_max}.dat"

pickle.dump(selection_model,open(path,"wb"))
joblib.dump(selection_model,path[:-4]+"_joblib_"+path[-4:])
print("start")

model2=LGBMClassifier()
model2=pickle.load(open(path,"rb"))
model2 = joblib.load(path[:-4]+"_joblib_"+path[-4:])

print("end")

