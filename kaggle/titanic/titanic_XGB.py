import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier,plot_importance
from sklearn.model_selection import train_test_split as tts


# train=pd.read_csv("kaggle/titanic/train.csv",index_col=0,header=0,encoding='cp949',sep=",")
train=pd.read_csv("kaggle/titanic/csv/train.csv",index_col=0,header=0,sep=",")
test=pd.read_csv("kaggle/titanic/csv/test.csv",index_col=0,header=0,encoding='cp949',sep=',')
gender_submission=pd.read_csv("./kaggle/titanic/csv/gender_submission.csv",index_col=0,header=0,encoding="cp949",sep=',')


# print(train.shape)

y= train.iloc[:,[0]]
x= train.iloc[:,[1,3,4,5,6,8]]


tmp = []
for each in train['Sex']:
    if each == 'female':
        tmp.append(1)
    elif each == 'male':
        tmp.append(0)
    else:
        tmp.append(np.nan)
        
train['Sex'] = tmp

y= train.iloc[:,[0]].values
x= train.iloc[:,[1,3,4,5,6,8]].values


x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,random_state=66)

#1.데이터 구성


#2) 모델

model  = XGBClassifier(n_estimators=1000)


#3) 트레이닝

model.fit(x_train,y_train,eval_metric = "rmse")

# 4) 평가 및 예측

y_pre = model.predict(x_test)
y_test= y_test.reshape(-1,)

print(y_test.shape)
print(y_pre.shape)

acc = model.score(x_test,y_test)

# y_pre= y_pre.reshape(-1,)

print(acc)

print(f"y_test[0:30]:{y_test[0:30]}")
print(f"y_pre[0:30]:{y_pre[0:30]}")

tmp = []
for each in test['Sex']:
    if each == 'female':
        tmp.append(1)
    elif each == 'male':
        tmp.append(0)
    else:
        tmp.append(np.nan)
        
test['Sex'] = tmp


index = np.array(test.index)

test =test.iloc[:,[0,2,3,4,5,7]].values

y_pre = model.predict(test)

print()
# y_pre = pd.Series(y_pre, name = 'Survived')

# print(test.index.shape)
# y_pre = y_pre.reshape(-1,)
# print(index)
# print(y_pre.shape)

submission = pd.DataFrame({
    "PassengerId": index,
    "Survived": y_pre
})


print(submission)

submission.to_csv(f'./kaggle/titanic/csv/{__file__[-(11+3):-3]}-{acc}.csv', index = False)

plot_importance(model)
plt.show()