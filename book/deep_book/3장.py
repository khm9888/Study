#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 필요한 모듈을 import합니다
import numpy
from sklearn.metrics import confusion_matrix

# 데이터를 저장합니다. 여기에서는 0이 양성을, 1이 음성을 나타냅니다
y_true = [0,0,0,1,1,1]
y_pred = [1,0,0,1,1,1]

# 변수 confmat에 y_true와 y_pred의 혼동 행렬을 저장하세요
confmat = confusion_matrix(y_true, y_pred)

# 결과를 출력합니다
print (confmat)


# In[2]:


# 적합율, 재현율, F
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

# 데이터를 저장합니다. 여기에서는 0이 양성, 1이 음성을 보여줍니다
y_true = [0,0,0,1,1,1]
y_pred = [1,0,0,1,1,1]

#y_true에는 정답 라벨을, y_pred에는 예측 결과의 라벨을 각각 전달합니다
print("Precision: %.3f" % precision_score(y_true, y_pred))
print("Recall: %.3f" % recall_score(y_true, y_pred))
print("F1: %.3f" % f1_score(y_true, y_pred))


# In[3]:


# 적합율, 재현율, F1
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score 

# 데이터를 저장합니다. 여기에서는 0이 음성, 1이 양성을 보여줍니다
y_true = [1,1,1,0,0,0]
y_pred = [0,1,1,0,0,0] 

# 적합율과 재현율을 미리 계산합니다
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred) 

# 다음 줄에 F1 점수의 정의식을 작성하세요
f1_score = 2 * (precision*recall) / (precision+recall)

print("F1: %.3f" % f1_score)


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# (1.)
#Iris 데이터 세트를 로드합니다
iris = datasets.load_iris()
# 3, 4번째의 특징을 추출합니다
X = iris.data[:, [2,3]]
# 클래스 라벨을 가져옵니다
y = iris.target

# (2.)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=0)

# (3.)
svc = svm.SVC(C=1, kernel='rbf', gamma=0.001)
svc.fit(X_train, y_train)

# (4.)
y_pred = svc.predict(X_test)
print ("Accuracy: %.2f"% accuracy_score(y_test, y_pred))


# In[ ]:




