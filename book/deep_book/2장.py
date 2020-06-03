#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 코드의 실행에 필요한 모듈을 읽습니다.
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Iris라는 데이터 세트를 읽습니다
iris = datasets.load_iris()
X = iris.data
y = iris.target

# X_train, X_test, y_train, y_test에 데이터를 저장합니다
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=0)
# 훈련 데이터와 테스트 데이터의 사이즈를 확인합니다
print ("X_train :", X_train.shape)
print ("y_train :", y_train.shape)
print ("X_test :", X_test.shape)
print ("y_test :", y_test.shape)


# In[3]:


# 코드의 실행에 필요한 모듈을 로드합니다
from sklearn import svm, datasets, cross_validation

# "Iris"라는 데이터 세트를 가져옵니다
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 머신 러닝 알고리즘 SVM을 사용합니다
svc = svm.SVC(C=1, kernel="rbf", gamma=0.001)

# 교차 검증법을 이용하여 점수를 요구합니다
# 내부에서는 X, y가 각각 X_train, X_test, y_train, y_test처럼 분할 처리됩니다
scores = cross_validation.cross_val_score(svc, X, y, cv=5)

# 학습 데이터와 테스트 데이터의 크기를 확인합니다
print (scores)
print ("평균 점수: ", scores.mean())


# In[ ]:




