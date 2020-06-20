#wine 퀄 classify 70 가장 우측이 wine quality
#wine 퀄 classify 70 가장 우측이 wine quality


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import Dense, Input,LSTM
from sklearn.metrics import r2_score,mean_squared_error as mse,accuracy_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC,SVC
from keras.utils import np_utils
from sklearn.decomposition import PCA
# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


#0. 데이터 전처리

wine_df = pd.read_csv("./data/winequality-white.csv",header=0,encoding="cp949",sep=";")

# count_data = wine_df.groupby("quality")["quality"]


# print(count_data)

# plt.plot(count_data)
# plt.show()


x= wine_df.drop("quality",axis=1)
y= wine_df["quality"]

newlist=[]

for i in y:
    if  i <=3:
        newlist+=[3]
    elif i>7:
        newlist+=[8]
    else:
        newlist+=[i]

y=newlist        
        
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)
        
print("-"*30+"RandomForestClassifier"+"-"*30)

model = RandomForestClassifier()

#3.훈련

model.fit(x_train,y_train)

#4.평가 및 예측

y_pre = model.predict(x_test)

score = model.score(x_test,y_test)
acc = accuracy_score(y_test,y_pre)


print(f"score:{score}")
print(f"r2:{r2_score(y_test,y_pre)}")
print(f"acc:{acc}")
print(f"y_test[0:100]:{y_test[0:100]}")
print(f"y_pre[0:100]:{y_pre[0:100]}")

print("-"*60)