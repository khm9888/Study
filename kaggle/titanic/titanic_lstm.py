import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Dense, Input, LSTM
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils



train=pd.read_csv("./kaggle/titanic/train.csv",index_col=0,header=0,encoding='cp949',sep=",")
test=pd.read_csv("./kaggle/titanic/test.csv",index_col=0,header=0,encoding='cp949',sep=',')
gender_submission=pd.read_csv("./kaggle/titanic/gender_submission.csv",index_col=0,header=0,encoding="cp949",sep=',')

# print(train)#1~801번까지 승객의 생존여부 다 있음 
# print(train.shape)# (891, 11)


# print(test)#892~1039번까지 생존여부 x
# print(test.shape) #(418, 10)

# print(gender_submission)#892~1039번까지 생존여부 o 
# print(gender_submission.shape)#(418, 1)

# #살아남은 사람들의 퍼센트 -1, 절대값-2

# f,ax=plt.subplots(1,2,figsize=(12,6))#f?

# train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.2f%%',ax=ax[0])
# ax[0].set_title('Survived')
# ax[0].set_ylabel('')

# sns.countplot('Survived',data=train,ax=ax[1])
# ax[1].set_title('Survived')
# plt.show()



# #나이별로 

# plt.title('Age')
# train['Age'].hist(bins=20,figsize=(18,8),grid=False)#bin-y값 간격
# plt.show()

# #그룹별로 묶음
# train.groupby('Pclass').mean()

# #각 값들의 상관 관계를 seaborn으로 표현

# plt.figure(figsize=(10, 10))
# sns.heatmap(train.corr(), linewidths=0.01, square=True,
#             annot=True, cmap=plt.cm.viridis, linecolor="white")
# plt.title('Correlation between features')
# plt.show()

# # 나이대와 좌석등급과 성별과 생존율 분석
# train['Age_cat'] = pd.cut(train['Age'], bins=[0, 10, 20, 50, 100], 
#                              include_lowest=True, labels=['baby', 'teenage', 'adult', 'old'])
# plt.figure(figsize=[12,4])
# plt.subplot(131)
# sns.barplot('Pclass', 'Survived', data=train)
# plt.subplot(132)
# sns.barplot('Age_cat', 'Survived', data=train)
# plt.subplot(133)
# sns.barplot('Sex', 'Survived', data=train)
# plt.subplots_adjust(top=1, bottom=0.1, left=0.10, right=1, hspace=0.5, wspace=0.5)
# plt.show()

# #실제 나이별 퍼센티지와, 생존자의 퍼센티지
# f,ax = plt.subplots(figsize=(12,6))
# g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], ax = ax, color="Blue", shade = True)
# g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax = g, color="Green", shade= True)
# g.set_xlabel("Age")
# g.set_ylabel("Frequency")
# g = g.legend(["Not Survived","Survived"])
# plt.show()

# # 남녀비율과, 각성별 생존자, 사망자수
# f,ax=plt.subplots(1,2,figsize=(12,6))
# sns.countplot('Sex',data=train, ax=ax[0])
# ax[0].set_title('Count of Passengers by Sex')

# sns.countplot('Sex',hue='Survived',data=train, ax=ax[1])
# ax[1].set_title('Sex:Survived vs Dead')
# plt.show()


tmp = []
for each in train['Sex']:
    if each == 'female':
        tmp.append(1)
    elif each == 'male':
        tmp.append(0)
    else:
        tmp.append(np.nan)

train['Sex'] = tmp

# train['survived'] = train['survived'].astype('float')
# train['pclass'] = train['pclass'].astype('float')
# train['sex'] = train['sex'].astype('float')
# train['sibsp'] = train['sibsp'].astype('float')
# train['parch'] = train['parch'].astype('float')
# train['fare'] = train['fare'].astype('float')

# train = train[train['Age'].notnull()]
# train = train[train['SibSp'].notnull()]
# train = train[train['Parch'].notnull()]
# train = train[train['Fare'].notnull()]


# print(train.shape)# (891, 11) -> #(714, 11)
# print(train)


test=pd.concat([test,gender_submission],axis=1)

# print(test)

tmp = []
for each in test['Sex']:
    if each == 'female':
        tmp.append(1)
    elif each == 'male':
        tmp.append(0)
    else:
        tmp.append(np.nan)

test['Sex'] = tmp

# test = test[test['Age'].notnull()]
# test = test[test['SibSp'].notnull()]
# test = test[test['Parch'].notnull()]
# test = test[test['Fare'].notnull()]


# print("135 - ",test.shape)#(418, 10) -> #(331, 10)

#1.데이터 구성

train.info() #확인

x_train = train.values[:,[1,3,4,8]]#테스트 해보기
# x_train = train.values[:,[1,3,4]]
y_train = train.values[:,[0]]
test.info() #확인

x_test = test.values[:,[0,2,3,7]]
# y_test = test.values[:,[10]]
y_test = test.values[:,[10]]


# print(y_train)
# print(y_test)

# print(gender_submission.shape)


#모델에 대한 구성. y값은 생존유무인지라 binary classify. 

#scaler
scaler = StandardScaler()
x_test = scaler.fit_transform(x_test)
x_train = scaler.fit_transform(x_train)

y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# #y값에 대한 전처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


x_train = x_train.reshape(-1,4,1)
x_test = x_test.reshape(-1,4,1)



#2) 모델

input1 = Input(shape=(4,1))
dense1 = LSTM(1000,activation="relu")(input1)
# dense1 = Dense(1000,activation="relu")(dense1)
dense1 = Dense(2,activation="sigmoid")(dense1)

model = Model(inputs = input1,outputs = dense1)

#3) 트레이닝

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])
model.fit(x_train,y_train,batch_size=10,epochs=30,validation_split=0.2,verbose=1)

# 4) 평가 및 예측

loss,acc = model.evaluate(x_test,y_test)
y_pre = model.predict(x_test)

y_test=np.argmax(y_test,axis=1)
y_pre=np.argmax(y_pre,axis=1)


# y_test= y_test.reshape(-1,)
# y_pre= y_pre.reshape(-1,)

print(f"loss:{loss}")
print(f"acc:{acc}")

print(f"y_test[0:30]:{y_test[0:30]}")
print(f"y_pre[0:30]:{y_pre[0:30]}")

# y_pre = pd.Series(y_pre, name = 'Survived')

# print(test.index.shape)
index = np.array(test.index)
y_pre = y_pre.reshape(-1,)
# print(index)
# print(y_pre.shape)
submission = pd.DataFrame({
    "PassengerId": index,
    "Survived": y_pre
})


print(submission)

submission.to_csv('submission_rf.csv', index = False)