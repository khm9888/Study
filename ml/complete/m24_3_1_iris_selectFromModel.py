from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split as tts
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import score_acc_score

iris = load_iris()

# print(type(iris))

# print(iris.keys())

x= iris.data
y= iris.target

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

xgb = XGBClassifier()

xgb.fit(x_train,y_train)

acc= xgb.score(x_test,y_test)

print(f"acc : {acc}")

thresholds = np.sort(xgb.feature_importances_)

max=-1
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
acc : 0.9
------------------------------
idx
0
thresh
0.017598107
select_score_acc
0.8231827111984282
------------------------------
------------------------------
idx
1
thresh
0.02607087
select_score_acc
0.8231827111984282
------------------------------
------------------------------
idx
2
thresh
0.33706376
select_score_acc
0.9472759226713533
------------------------------
------------------------------
idx
3
thresh
0.6192673
select_score_acc
0.8993288590604027
------------------------------
2
0.9472759226713533
'''