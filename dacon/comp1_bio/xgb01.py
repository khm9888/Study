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

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#2.모델 구성-xgb
from xgboost import XGBRegressor,plot_importance

xgb0 = XGBRegressor()
xgb1 = XGBRegressor()
xgb2 = XGBRegressor()
xgb3 = XGBRegressor()

y_train0=y_train[:,0]
y_train1=y_train[:,1]
y_train2=y_train[:,2]
y_train3=y_train[:,3]

y_test0 = y_test[:,0]
y_test1 = y_test[:,1]
y_test2 = y_test[:,2]
y_test3 = y_test[:,3]

xgb0.fit(x_train,y_train0)
xgb1.fit(x_train,y_train1)
xgb2.fit(x_train,y_train2)
xgb3.fit(x_train,y_train3)

y_pre0=xgb0.predict(x_test)
y_pre1=xgb1.predict(x_test)
y_pre2=xgb2.predict(x_test)
y_pre3=xgb3.predict(x_test)

r2_0= xgb0.score(x_test,y_test0)
r2_1= xgb1.score(x_test,y_test1)
r2_2= xgb2.score(x_test,y_test2)
r2_3= xgb3.score(x_test,y_test3)

mae_0= mae(y_test0,y_pre0)
mae_1= mae(y_test1,y_pre1)
mae_2= mae(y_test2,y_pre2)
mae_3= mae(y_test3,y_pre3)

# m=list(mae_0,mae_1,mae_2,mae_3)
mae_result = (mae_0+mae_1+mae_2+mae_3)/4
print("__file__")
print(__file__)

print(f"mae_0 : {mae_0}")
print(f"mae_1 : {mae_1}")
print(f"mae_2 : {mae_2}")
print(f"mae_3 : {mae_3}")

print(f"mae_result : {mae_result}")

print(f"r2_0 : {r2_0}")
print(f"r2_1 : {r2_1}")
print(f"r2_2 : {r2_2}")
print(f"r2_3 : {r2_3}")

###################################################################################


# print("*"*30,"truly","*"*30)

y_pre0=xgb0.predict(test)
y_pre1=xgb1.predict(test)
y_pre2=xgb2.predict(test)
y_pre3=xgb3.predict(test)

y_pre=[y_pre0,y_pre1,y_pre2,y_pre3]
y_pre=np.array(y_pre)
y_pre=y_pre.transpose()

numbering=2
submission = pd.DataFrame(y_pre,np.arange(10000,20000))
submission.to_csv(f"dacon\comp1_bio\csv\sample_submission_{__file__[-8:-3]}_{numbering}.csv", index = True, header=['hhb','hbo2','ca','na'],index_label='id')

# submit = pd.DataFrame({})

# mae_0 : 1.0570167305499316
# mae_1 : 0.6979092119789123
# mae_2 : 2.0381685493278505
# mae_3 : 1.4167864456421135
# mae_result : 1.3024702343747019
# r2_0 : 0.7734095048159794
# r2_1 : 0.24005958464434862
# r2_2 : 0.28260562671886913
# r2_3 : 0.09562861519521182