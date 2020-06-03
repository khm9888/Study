'''
id : 구분자
rho : 측정 거리 (단위: mm)
src : 광원 스펙트럼 (650 nm ~ 990 nm)
dst : 측정 스펙트럼 (650 nm ~ 990 nm)
hhb : 디옥시헤모글로빈 농도
hbo2 : 옥시헤모글로빈 농도
ca : 칼슘 농도
na : 나트륨 농도
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM,Input
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from keras.callbacks import EarlyStopping,ModelCheckpoint
e_stop = EarlyStopping(monitor='loss',patience=5,mode='auto')

m_check = ModelCheckpoint(filepath=".\keras_prac\model\{epoch:02d}--{val_loss:.4f}.hdf5", monitor = 'val_loss',save_best_only=True)

train = pd.read_csv('./dacon/data/train.csv',index_col=0)
test = pd.read_csv('./dacon/data/test.csv', index_col=0)
submission = pd.read_csv('./dacon/data/sample_submission.csv')

train_dst = train.filter(regex='_dst$', axis=1)
test_dst = test.filter(regex='_dst$', axis=1)
test_dst.head(1)

# print(train_dst)

train_dst = train_dst.interpolate(methods='linear', axis=1)
test_dst = test_dst.interpolate(methods='linear', axis=1)

# 스팩트럼 데이터에서 보간이 되지 않은 값은 0으로 일괄 처리한다.

train_dst.fillna(0, inplace=True) 
test_dst.fillna(0, inplace=True)

# print(test_dst.head())
# 
train.update(train_dst) # 보간한 데이터를 기존 데이터프레임에 업데이트 한다.
test.update(test_dst)
# 


# print(df.shape)
# col = range(650,1000,10)

# x_train = df

#데이터 분석

train.filter(regex='_src$',axis=1)[16:20].T.plot()
plt.figure(figsize=(4,12))
train.filter(regex='_dst$',axis=1)[16:20].T.plot()
sns.heatmap(train.corr().loc['rho':'990_dst','hhb':].abs())
sns.heatmap(train.corr().loc['650_dst':'990_dst','650_src':'990_src'].abs())
# sns.heatmap(train.corr().loc['650_src':'990_src','650_dst':'990_dst'].abs().plot())
plt.show()



# print(train.head())
# test.head()
# msno.matrix(train)
# msno.matrix(test)
# print(train.isnull().sum()[train.isnull().sum().values > 0])
# print(train.isnull().sum()[train.isnull().sum().values > 0].index)

# train_dst.head().T.plot()
# test_dst.head().T.plot()
# plt.show()
# print(train_dst.head())

# print(test_dst.head())


x_train = train.loc[:,'rho':'990_dst']
y_train = train.loc[:,'hhb':]
sclar = StandardScaler()
sclar.fit(x_train)
x_train = sclar.transform(x_train)
test =sclar.transform(test)

# print(x_train.shape, y_train.shape)
skf = KFold(n_splits=5,shuffle=True)
accuracy = []
for t, v in skf.split(x_train):
    print(t)
    x_t , y_t = x_train[t], y_train.iloc[t]

    model = Sequential()
    model.add(Dense(10000,input_shape=(71,),activation='relu'))

    model.add(Dense(1000,activation='relu'))
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(4))

    
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

    # 학습 데이터를 이용해서 학습
    model.fit(x_t, y_t, epochs=100, batch_size=20)

    # 테스트 데이터를 이용해서 검증
    k_accuracy = '%.4f' % (model.evaluate(x_train[v], y_train[v])[1])
    accuracy.append(k_accuracy)


print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
