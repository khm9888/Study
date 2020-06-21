from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split as tts,GridSearchCV
import numpy as np
from xgboost import XGBRegressor,plot_importance
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

boston = load_boston()

# print(type(boston))

# print(boston.keys())

x= boston.data
y= boston.target

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

xgb = XGBRegressor()

xgb.fit(x_train,y_train)

plot_importance(xgb)
plt.title(1)
plt.show()

r2= xgb.score(x_test,y_test)

print("-"*10,"start","-"*10)
print(f"r2 : {r2}")

thresholds = np.sort(xgb.feature_importances_)



selection=SelectFromModel(xgb,threshold=thresholds[4],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.

selection_x_train = selection.transform(x_train)
selection_x_test = selection.transform(x_test)

n_estimators = 1100
learning_rate = 0.01
colsample_bytree=0.3#우승모델은 0.6~0.9
colsample_bylevel=0.9#우승모델은 0.6~0.9
max_depth = 8
n_jobs=-1

parameters = {
    "n_estimators" : np.array([1500]),
    # "learning_rate" : np.arange(0.005,0.031,0.002),
    "learning_rate" : np.array([0.008]),
    # "colsample_bytree": np.array([0.9]),
    # "colsample_bylevel": np.array([0.3,1,0.1])
    "max_depth" : [6,7,8,9]
}

xgb2 = XGBRegressor()

xgb2.fit(selection_x_train,y_train)

print("__file__")
print(__file__)
print("xgb2")
xgb2.fit(selection_x_train,y_train)
y_pre = xgb2.predict(selection_x_test)
r2 = r2_score(y_pre,y_test)
print("r2")
print(r2)

model = GridSearchCV(xgb2,parameters,cv=5,n_jobs=-1)


model.fit(selection_x_train,y_train)

y_pre = model.predict(selection_x_test)
r2 = r2_score(y_pre,y_test)

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
print("r2")
print(r2)
print("-"*30)

# print("y_pre")
# print(y_pre)

'''
4
0.9299836019002435
PS D:\Study> se
'''

'''
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
'''
'''
model.best_params_
{'colsample_bylevel': 0.30000000000000004, 'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 1100}
------------------------------
------------------------------
d:\Study\ml\m24_1_2_boston_select_complete.py
r2
0.9240597166946359
'''