from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split as tts,GridSearchCV
import numpy as np
from xgboost import XGBClassifier,plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

# print(type(cancer))

# print(cancer.keys())

x= cancer.data
y= cancer.target

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

xgb = XGBClassifier()

xgb.fit(x_train,y_train)

# plot_importance(xgb)
# plt.title(1)
# plt.show()

score_acc= xgb.score(x_test,y_test)
print("-"*10,"start","-"*10)
print(f"score_acc : {score_acc}")

thresholds = np.sort(xgb.feature_importances_)


selection=SelectFromModel(xgb,threshold=thresholds[22],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.

selection_x_train = selection.transform(x_train)
selection_x_test = selection.transform(x_test)

# n_estimators = 200
# learning_rate = 0.02
# colsample_bytree=0.8#우승모델은 0.6~0.9
# colsample_bylevel=0.6#우승모델은 0.6~0.9
# max_depth = 4
# n_jobs=-1

# parameters = {
#     "n_estimators" : np.arange(100,301,100),
#     "learning_rate" : np.arange(0.01,0.03,0.01),
#     "colsample_bytree":np.arange(0.6,1,0.1),
#     "colsample_bylevel":np.arange(0.6,1,0.1),
#     "max_depth" : [4,5,6]
# }


parameters = {
    "n_estimators" : np.arange(50,251,50),
    "learning_rate" : np.arange(0.02,0.031,0.01),
    "colsample_bytree": np.array([0.8,0.9,1]),
    "colsample_bylevel":np.arange(0.1,0.41,0.1),
    "max_depth" : [3,4,5]
}


# xgb2 = XGBClassifier (max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, colsample_bytree=colsample_bytree,
#                       colsample_bylevel=colsample_bylevel,n_jobs=n_jobs)
xgb2 = XGBClassifier()

print("__file__")
print(__file__)
print("xgb2")
xgb2.fit(selection_x_train,y_train)
y_pre = xgb2.predict(selection_x_test)
score_acc = r2_score(y_pre,y_test)
print("score_acc")
print(score_acc)
# plot_importance(xgb2)
# plt.show()

model = GridSearchCV(xgb2,parameters,cv=5,n_jobs=-1)


model.fit(selection_x_train,y_train)

y_pre = model.predict(selection_x_test)
score_acc = r2_score(y_pre,y_test)

print("-"*30)
print("__file__")
print(__file__)
print("model.best_estimator_")
print(model.best_estimator_)
print("-"*30)
print("model.best_params_")
print(model.best_params_)
print("-"*30)

print("-"*30)
print(__file__)
print("score_acc")
print(score_acc)
print("-"*30)

# print("y_pre")
# print(y_pre)



'''
---------- start ----------
score_acc : 0.9736842105263158
__file__
d:\Study\ml\m24_2_2_cancer_select_complete.py
xgb2
score_acc
0.8844594594594595
------------------------------
__file__
d:\Study\ml\m24_2_2_cancer_select_complete.py
model.best_estimator_
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.4,
              colsample_bynode=1, colsample_bytree=0.8, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.02, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=-1, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
------------------------------
model.best_params_
{'colsample_bylevel': 0.4, 'colsample_bytree': 0.8, 'learning_rate': 0.02, 'max_depth': 4, 'n_estimators': 100}
------------------------------
------------------------------
d:\Study\ml\m24_2_2_cancer_select_complete.py
score_acc
0.8844594594594595
------------------------------

'''