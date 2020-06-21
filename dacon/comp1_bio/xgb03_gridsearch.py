import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score,mean_absolute_error as mae
# from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential,Model
# from keras.layers import Dense,Input,Dropout,LSTM
from sklearn.model_selection import GridSearchCV


#0~1 데이터 전처리, 데이터 입력

train = pd.read_csv("./dacon/comp1_bio/train.csv",index_col=0,header=0,encoding="cp949")
test = pd.read_csv("./dacon/comp1_bio/test.csv",index_col=0,header=0,encoding="cp949")
submission = pd.read_csv("./dacon/comp1_bio/csv/sample_submission.csv",index_col=0,header=0,encoding="cp949")

# for i in range(len()):
    
    
# print(train.shape)#(10000, 75) train,test
# print(test.shape)#(10000, 71) x_predict
# print(submission.shape)#(10000, 4) y_predict


# print(train.isnull().sum())

# train = train.interpolate()#보간법 // 선형보간

# print(dir(train))
# print(train.isnull().sum())


test = test.interpolate() #보간법 // 선형보간

# print(type(train))

# train = train.fillna(method="pad",axis=1)
test = test.fillna(method="pad",axis=1)


# print(test.info())

test=test.values


# print(train.shape)

x = train.values[:,:-4]
y = train.values[:,-4:]



# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.7,
                                                    random_state=66)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#2.모델 구성-xgb
from xgboost import XGBRegressor,plot_importance

xgb_0 = XGBRegressor()
xgb_1 = XGBRegressor()
xgb_2 = XGBRegressor()
xgb_3 = XGBRegressor()

y_train_0=y_train[:,0]
y_train_1=y_train[:,1]
y_train_2=y_train[:,2]
y_train_3=y_train[:,3]

y_test_0 = y_test[:,0]
y_test_1 = y_test[:,1]
y_test_2 = y_test[:,2]
y_test_3 = y_test[:,3]


###################################################################################

parameters_0 = {
    "n_estimators" : np.arange(700,701,100),
    # "learning_rate" : np.arange(0.01,0.03,0.01),
    "colsample_bytree":np.arange(0.6,0.9,0.1),
    # "colsample_bylevel":np.arange(0.6,1,0.1),
    # "max_depth" : [4,5,6]
}

xgb_0 = XGBRegressor ()
#트리구조는 전처리를 안 해도 됨. - pipeline,
#결측치 제거 안 해도 됨 - 보간법 적용.
#해주긴 하지만, 보간법 적용하는게 나을 수 있음.
#여러모델보단 느림. 앙상블이라.
xgb_0.fit(x_train,y_train_0)

xgb_0 = GridSearchCV(xgb_0,parameters_0,cv=5,n_jobs=-1)

# m=list(mae_0,mae_1,mae_2,mae_3)


print("__file__")
print(__file__)


xgb_0.fit(x_train,y_train_0)
xgb_1.fit(x_train,y_train_1)
xgb_2.fit(x_train,y_train_2)
xgb_3.fit(x_train,y_train_3)

y_pre_0=xgb_0.predict(x_test)
y_pre_1=xgb_1.predict(x_test)
y_pre_2=xgb_2.predict(x_test)
y_pre_3=xgb_3.predict(x_test)

r2_0= xgb_0.score(x_test,y_test_0)
r2_1= xgb_1.score(x_test,y_test_1)
r2_2= xgb_2.score(x_test,y_test_2)
r2_3= xgb_3.score(x_test,y_test_3)

mae_0= mae(y_test_0,y_pre_0)
mae_1= mae(y_test_1,y_pre_1)
mae_2= mae(y_test_2,y_pre_2)
mae_3= mae(y_test_3,y_pre_3)

# m=list(mae_0,mae_1,mae_2,mae_3)


print("__file__")
print(__file__)

mae_result = (mae_0+mae_1+mae_2+mae_3)/4
r2_result = (r2_0+r2_1+r2_2+r2_3)/4

print(xgb_0.best_params_)
print()
print(f"mae_0 : {mae_0}")
print(f"mae_1 : {mae_1}")
print(f"mae_2 : {mae_2}")
print(f"mae_3 : {mae_3}")

print(f"mae_result : {mae_result}")

print(f"r2_0 : {r2_0}")
print(f"r2_1 : {r2_1}")
print(f"r2_2 : {r2_2}")
print(f"r2_3 : {r2_3}")

print(f"r2_result : {r2_result}")


print("-"*30)
print("__file__")
print(__file__)
print("xgb_0.best_estimator_")
print(xgb_0.best_estimator_)
print("-"*30)
print("xgb_0.best_params_")
print(xgb_0.best_params_)
print("-"*30)


# mae_0 : 1.062883681857586
# mae_1 : 0.6896557573413847
# mae_2 : 1.9638330566596984
# mae_3 : 1.398394598261118
# mae_result : 1.2786917735299468
# r2_0 : 0.768523796053417
# r2_1 : 0.26151210804907765
# r2_2 : 0.28713042620602325
# r2_3 : 0.1269457040690386
# r2_result : 0.3610280085943891