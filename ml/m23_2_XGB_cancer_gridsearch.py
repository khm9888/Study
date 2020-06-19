#과적합 방지
#1.훈련데이터 늘린다.
#2.feature를 줄인다.
#3.regularization

from xgboost import XGBClassifier,plot_importance
# from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import GridSearchCV
import numpy as np

dataset = load_breast_cancer()
# print(type(load_boston())) #<class 'sklearn.utils.Bunch'>
# print(print(dataset.keys()))

x= dataset.data
y= dataset.target

print(type(x)) #<class 'numpy.ndarray'>

print("x.shape")
print(x.shape)

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test  = tts(x,y,train_size=0.8)

n_estimators = 100
learning_rate = 0.01
colsample_bytree=0.9#우승모델은 0.6~0.9
colsample_bylevel=0.9#우승모델은 0.6~0.9
max_depth = 5
n_jobs=-1

parameters = {
    "n_estimators" : np.arange(100,301,100),
    "learning_rate" : np.arange(0.01,0.03,0.01),
    "colsample_bytree":np.arange(0.6,1,0.1),
    "colsample_bylevel":np.arange(0.6,1,0.1),
    "max_depth" : [4,5,6]
}


xgb = XGBClassifier (max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, colsample_bytree=colsample_bytree,
                      colsample_bylevel=colsample_bylevel,n_jobs=n_jobs)

#트리구조는 전처리를 안 해도 됨. - pipeline,
#결측치 제거 안 해도 됨 - 보간법 적용.
#해주긴 하지만, 보간법 적용하는게 나을 수 있음.
#여러모델보단 느림. 앙상블이라.

xgb.fit(x_train,y_train)

model = GridSearchCV(xgb,parameters,cv=5,n_jobs=-1)

model.fit(x_train,y_train)

r2=model.score(x_test,y_test)

print("-"*30)
print("__file__")
print(__file__)
print("model.best_estimator_")
print(model.best_estimator_)
print("-"*30)
print("model.best_params_")
print(model.best_params_)
print("-"*30)

print(xgb.feature_importances_)


plot_importance(xgb)
plt.show()


'''
__file__
d:\Study\ml\m23_2_XGB_cancer_gridsearch.py
model.best_estimator_
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.02, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=300, n_jobs=-1, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
------------------------------
model.best_params_
{'colsample_bylevel': 0.6, 'colsample_bytree': 0.6, 'learning_rate': 0.02, 'max_depth': 4, 'n_estimators': 300}
------------------------------
'''
