from sklearn.model_selection import train_test_split as tts
import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel

# 1.회귀
# 2.이진분류
# 3.다중분류

# eval 에 "loss"와 다른 지표 1개 더 추가
# earlystopping 적용
# plot 으로 그릴 것

# 4. 결과는 주석 하단

dataset = load_boston()

x= dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

xgb = XGBRegressor(n_estimators=100,learning_rate = 0.1,n_jobs=-1)

xgb.fit(x_train,y_train,verbose=True, eval_metric=["logloss","rmse"],
        eval_set=[(x_train, y_train)],early_stopping_rounds=20)


#rmse,mae,logloss,error,auc


y_pre = xgb.predict(x_test)

r2 = r2_score(y_test,y_pre)
score = xgb.score(x_test,y_test)
results = xgb.evals_result()
print(type(results))
print(__file__)
print(results)
print("r2")
print(r2)
print("score")
print(score)

# fig, ax = plt.subplots()

# epochs = len(results["validation_0"]["logloss"])
# x_axis = range(epochs)
# ax.plot(x_axis,results["validation_0"]["logloss"],label="Train")
# ax.plot(x_axis,results["validation_1"]["logloss"],label="Test")
# ax.legend()

# plt.ylabel("logloss")
# plt.show()

# fig, ax = plt.subplots()

# epochs = len(results["validation_0"]["rmse"])
# x_axis = range(epochs)
# ax.plot(x_axis,results["validation_0"]["rmse"],label="Train")
# ax.plot(x_axis,results["validation_1"]["rmse"],label="Test")
# ax.legend()

# plt.ylabel("rmse")
# plt.show()


#6)selectFromModel

thresholds = np.sort(xgb.feature_importances_)

idx_max = -1
max = r2

for idx,thresh in enumerate(thresholds):
    #데이터 전처리
    selection = SelectFromModel(xgb,threshold=thresh,prefit=True)
    #1)데이터입력
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    #2)모델구성
    selection_model = XGBRegressor(n_estimators=100,learning_rate = 0.1,n_jobs=-1)
    
    #3)훈련
    selection_model.fit(selection_x_train,y_train,verbose=False, eval_metric=["logloss","rmse"],eval_set=[(selection_x_train, y_train), (selection_x_test,y_test)],early_stopping_rounds=20)
    
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

print("idx_max")
print(idx_max)
print("max")
print(max)

selection = SelectFromModel(xgb,threshold=thresholds[idx_max],prefit=True)
selection_x_train = selection.transform(x_train)
selection_x_test = selection.transform(x_test)
#2)모델구성
selection_model = XGBRegressor(n_estimators=100,learning_rate = 0.1,n_jobs=-1)

#3)훈련
selection_model.fit(selection_x_train,y_train,verbose=False, eval_metric=["logloss","rmse"],eval_set=[(selection_x_train, y_train), (selection_x_test,y_test)],early_stopping_rounds=20)
path=f"./model/xgb_save/{__file__[-24:-3]}-idx{idx_max}.dat"
selection_model.save_model(path)
'''
r2
0.9328556062354909
score
0.9328556062354909
idx
0
r2
0.9328556062354909
idx
1
r2
0.932501384781691
idx
2
r2
0.9326566772692234
idx
3
r2
0.9339682837594144
idx
4
r2
0.9319608339462506
idx
5
r2
0.937173713298372
idx
6
r2
0.9355646841659825
idx
7
r2
0.940174120907745
idx
8
r2
0.9281739196057937
idx
9
r2
0.9200432278681383
idx
10
r2
0.8906439287785177
idx
11
r2
0.8124580017644472
idx
12
r2
0.6826341533338622
idx_max
7
max
0.940174120907745
'''
