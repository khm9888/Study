import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier,plot_importance
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_selection import SelectFromModel

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

# model.feature_importance_
thresholds = np.sort(model.feature_importances_)
score_acc = model.score(x_test,y_test)

max=-1
idx_max=score_acc
print(max)
print(idx_max)
for idx,thresh in enumerate(thresholds):
    selection = SelectFromModel(model,threshold=thresh,prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    
    selection_x_train = selection.transform(x_train)
    selection_x_test = selection.transform(x_test)
    # print(selection_x_train)

    selection_model =XGBClassifier()
    selection_model.fit(selection_x_train,y_train)
    
    y_pre = selection_model.predict(selection_x_test)
    select_score_acc = selection_model.score(selection_x_test,y_test)
    if select_score_acc>=max:
        max=select_score_acc
        idx_max=idx
    print("-"*30)
    print("idx")
    print(idx)
    print("thresh")
    print(thresh)
    print("select_score_acc")
    print(select_score_acc)
    print("-"*30)


print(idx_max)
print(max)

# 4) 평가 및 예측

selection = SelectFromModel(model,threshold=thresholds[0],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.

selection_x_train = selection.transform(x_train)
selection_x_test = selection.transform(x_test)
# print(selection_x_train)

selection_model =XGBClassifier()
selection_model.fit(selection_x_train,y_train)

y_pre = selection_model.predict(selection_x_test)
acc = selection_model.score(selection_x_test,y_test)
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
selection_test = selection.transform(test)

y_pre = selection_model.predict(selection_test)

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