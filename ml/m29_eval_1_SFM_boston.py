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
        eval_set=[(x_train, y_train), (x_test,y_test)],early_stopping_rounds=20)

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
    x_train = selection.transform(x_train)
    x_test = selection.transform(x_test)
    #2)모델구성
    # 이미 앞에 입력해서 생략
    
    #3)훈련
    xgb.fit(x_train,y_train,verbose=False, eval_metric=["logloss","rmse"],eval_set=[(x_train, y_train), (x_test,y_test)],early_stopping_rounds=20)
    
    #4)평가 및 예측
    y_pre = xgb.predict(x_test)
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