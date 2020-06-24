from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split as tts
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score as acc_score
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor

# 1.회귀
# 2.이진분류
# 3.다중분류

# eval 에 "loss"와 다른 지표 1개 더 추가
# earlystopping 적용
# plot 으로 그릴 것

# 4. 결과는 주석 하단

dataset = load_iris()

x=dataset.data
y=dataset.target

#1)데이터 입력

x_train,x_test,y_train,y_test = tts(x,y,train_size =0.8, random_state = 66)

#2)모델구성

xgb = XGBClassifier(n_estimators=1000,learning_rate=0.1,objective='multi:softmax',n_jobs=-1)

#3)훈련
xgb.fit(x_train,y_train,verbose=True,eval_metric=["mlogloss","merror"],eval_set=[(x_train,y_train),(x_test,y_test)],early_stopping_rounds=20)

#4)평가 및 예측

y_pre = xgb.predict(x_test)
acc = acc_score(y_test,y_pre)
print(__file__)
print("acc")
print(acc)
results = xgb.evals_result()

# #5) 그래프화

# epochs = len(results["validation_0"]['mlogloss'])
# x_axis = range(epochs)
# plt.plot(x_axis,results["validation_0"]['mlogloss'],label="Train")
# plt.plot(x_axis,results["validation_1"]['mlogloss'],label="Test")
# plt.show()


# epochs = len(results["validation_0"]['merror'])
# x_axis = range(epochs)
# plt.plot(x_axis,results["validation_0"]['merror'],label="Train")
# plt.plot(x_axis,results["validation_1"]['merror'],label="Test")
# plt.show()

'''
acc
1.0
'''

#6)selectFromModel

thresholds = np.sort(xgb.feature_importances_)

idx_max = -1
max = acc

for idx,thresh in enumerate(thresholds):
    #데이터 전처리
    selection = SelectFromModel(xgb,threshold=thresh,prefit=True)
    #1)데이터입력
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    #2)모델구성
    selection_model = XGBClassifier(n_estimators=100,learning_rate = 0.1,n_jobs=-1)
    
    #3)훈련
    selection_model.fit(selection_x_train,y_train,verbose=False,eval_metric=["mlogloss","merror"],eval_set=[(selection_x_train,y_train),(selection_x_test,y_test)],early_stopping_rounds=20)
    #4)평가 및 예측
    y_pre = selection_model.predict(selection_x_test)
    acc = acc_score(y_test,y_pre)
    print("idx")
    print(idx)
    print("acc")
    print(acc)
    if max<=acc:
        max=acc
        idx_max=idx

print()
print("idx_max")
print(idx_max)
print("max")
print(max)

selection = SelectFromModel(xgb,threshold=thresholds[idx_max],prefit=True)
selection_x_train = selection.transform(x_train)
selection_x_test = selection.transform(x_test)
#2)모델구성
selection_model = XGBClassifier(n_estimators=100,learning_rate = 0.1,n_jobs=-1)

#3)훈련
selection_model.fit(selection_x_train,y_train,verbose=False,eval_metric=["mlogloss","merror"],eval_set=[(selection_x_train,y_train),(selection_x_test,y_test)],early_stopping_rounds=20)
path=f"./model/xgb_save/{__file__[-(24-2):-3]}-idx{idx_max}.dat"
selection_model.save_model(path)

'''
acc
1.0
idx
0
acc
1.0
idx
1
acc
1.0
idx
2
acc
1.0
idx
3
acc
0.9333333333333333

idx_max
2
max
1.0
'''