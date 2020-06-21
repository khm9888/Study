from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split as tts
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import score_acc_score

cancer = load_breast_cancer()

# print(type(cancer))

# print(cancer.keys())

x= cancer.data
y= cancer.target

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

xgb = XGBClassifier()

xgb.fit(x_train,y_train)

score_acc= xgb.score(x_test,y_test)

print(f"score_acc : {score_acc}")

thresholds = np.sort(xgb.feature_importances_)
print(thresholds)


max=score_acc
idx_max=-1

for idx ,thresh in enumerate(thresholds):
    selection = SelectFromModel(xgb,threshold=thresh,prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    
    selection_x_train = selection.transform(x_train)
    
    selection_x_test = selection.transform(x_test)
    # print(selection_x_train)

    selection_model =XGBClassifier()
    selection_model.fit(selection_x_train,y_train)
    
    y_pre = selection_model.predict(selection_x_test)
    select_score_acc = score_acc_score(y_pre,y_test)
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
    
    # print("y_pre")
    # print(y_pre)

print(idx_max)
print(max)

'''
score_acc : 0.9736842105263158
------------------------------
idx
0
thresh
0.0
select_score_acc
0.8844594594594595
------------------------------
------------------------------
idx
1
thresh
0.0
select_score_acc
0.8844594594594595
------------------------------
------------------------------
idx
2
thresh
0.00037144974
select_score_acc
0.8844594594594595
------------------------------
------------------------------
idx
3
thresh
0.002333931
select_score_acc
0.8844594594594595
------------------------------
------------------------------
idx
4
thresh
0.0027849793
select_score_acc
0.8844594594594595
------------------------------
------------------------------
idx
5
thresh
0.0028118356
select_score_acc
0.8844594594594595
------------------------------
------------------------------
idx
6
thresh
0.0032604332
select_score_acc
0.8844594594594595
------------------------------
------------------------------
idx
7
thresh
0.003402722
select_score_acc
0.8844594594594595
------------------------------
------------------------------
idx
8
thresh
0.0036917944
select_score_acc
0.8441025641025641
------------------------------
------------------------------
idx
9
thresh
0.0043062596
select_score_acc
0.8441025641025641
------------------------------
------------------------------
idx
10
thresh
0.0050556003
select_score_acc
0.8844594594594595
------------------------------
------------------------------
idx
11
thresh
0.005134485
select_score_acc
0.8844594594594595
------------------------------
------------------------------
idx
12
thresh
0.005499401
select_score_acc
0.8441025641025641
------------------------------
------------------------------
idx
13
thresh
0.005847498
select_score_acc
0.8441025641025641
------------------------------
------------------------------
idx
14
thresh
0.006394118
select_score_acc
0.8441025641025641
------------------------------
------------------------------
idx
15
thresh
0.007691835
select_score_acc
0.8869047619047619
------------------------------
------------------------------
idx
16
thresh
0.0077531096
select_score_acc
0.8869047619047619
------------------------------
------------------------------
idx
17
thresh
0.009037063
select_score_acc
0.9238222519211493
------------------------------
------------------------------
idx
18
thresh
0.011710226
select_score_acc
0.9238222519211493
------------------------------
------------------------------
idx
19
thresh
0.013685605
select_score_acc
0.9238222519211493
------------------------------
------------------------------
idx
20
thresh
0.014204989
select_score_acc
0.9220512820512821
------------------------------
------------------------------
idx
21
thresh
0.01813928
select_score_acc
0.881578947368421
------------------------------
------------------------------
idx
22
thresh
0.022859031
select_score_acc
0.881578947368421
------------------------------
------------------------------
idx
23
thresh
0.023654878
select_score_acc
0.9220512820512821
------------------------------
------------------------------
idx
24
thresh
0.03333857
select_score_acc
0.881578947368421
------------------------------
------------------------------
idx
25
thresh
0.06629944
select_score_acc
0.7970085470085471
------------------------------
------------------------------
idx
26
thresh
0.09745205
select_score_acc
0.8441025641025641
------------------------------
------------------------------
idx
27
thresh
0.115862854
select_score_acc
0.7599157599157599
------------------------------
------------------------------
idx
28
thresh
0.22248562
select_score_acc
0.5877034358047017
------------------------------
------------------------------
idx
29
thresh
0.28493083
select_score_acc
0.4722222222222223
------------------------------
-1
0.9736842105263158'''