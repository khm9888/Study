from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split as tts
import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import time

start=time.time()
boston = load_boston()

# print(type(boston))

# print(boston.keys())

x= boston.data
y= boston.target

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

xgb = XGBRegressor(n_jobs=-1)

xgb.fit(x_train,y_train)

r2= xgb.score(x_test,y_test)

print(f"r2 : {r2}")

thresholds = np.sort(xgb.feature_importances_)

max=-1
idx_max=-1

for idx ,thresh in enumerate(thresholds):
    selection = SelectFromModel(xgb,threshold=thresh,prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    
    selection_x_train = selection.transform(x_train)
    
    selection_x_test = selection.transform(x_test)
    # print(selection_x_train)

    selection_model =XGBRegressor(n_jobs=-1)
    selection_model.fit(selection_x_train,y_train)
    
    y_pre = selection_model.predict(selection_x_test)
    select_r2 = r2_score(y_pre,y_test)
    if select_r2>=max:
        max=select_r2
        idx_max=idx
    print("-"*30)
    print("idx")
    print(idx)
    print("thresh")
    print(thresh)
    print("select_r2")
    print(select_r2)
    print("-"*30)
    
    # print("y_pre")
    # print(y_pre)

print(idx_max)
print(max)
print(time.time()-start)
# 2.3427329063415527 (기본값)

# 2.3447279930114746 (-1)

'''
r2 : 0.9221188544655419
------------------------------
idx
0
thresh
0.0013415267
select_r2
0.9190818305521244
------------------------------
------------------------------
idx
1
thresh
0.0036337192
select_r2
0.9180023433459444
------------------------------
------------------------------
idx
2
thresh
0.012031149
select_r2
0.9178219368110624
------------------------------
------------------------------
idx
3
thresh
0.012204577
select_r2
0.919012105666861
------------------------------
------------------------------
idx
4
thresh
0.014479355
select_r2
0.9299836019002435
------------------------------
------------------------------
idx
5
thresh
0.014791191
select_r2
0.9210097578362715
------------------------------
------------------------------
idx
6
thresh
0.017543204
select_r2
0.9120217586319334
------------------------------
------------------------------
idx
7
thresh
0.030416546
select_r2
0.9217295973499727
------------------------------
------------------------------
idx
8
thresh
0.04246345
select_r2
0.9129276797015515
------------------------------
------------------------------
idx
9
thresh
0.051825397
select_r2
0.9196473612891426
------------------------------
------------------------------
idx
10
thresh
0.06949984
select_r2
0.915998235777641
------------------------------
------------------------------
idx
11
thresh
0.30128643
select_r2
0.6609210304673123
------------------------------
------------------------------
idx
12
thresh
0.42848358
select_r2
0.2531320229972658
------------------------------
4
0.9299836019002435
PS D:\Study> se
'''