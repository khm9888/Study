from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split as tts
import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

boston = load_boston()

# print(type(boston))

# print(boston.keys())

x= boston.data
y= boston.target

x_train, x_test, y_train, y_test = tts(x,y,train_size=0.8,
                                                    random_state=66)

xgb = XGBRegressor()

xgb.fit(x_train,y_train)

r2= xgb.score(x_test,y_test)

print(f"r2 : {r2}")

thresholds = np.sort(xgb.feature_importances_)

for idx ,thresh in enumerate(thresholds):
    selection = SelectFromModel(xgb,threshold=thresh,prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    
    selection_x_train = selection.transform(x_train)
    
    selection_x_test = selection.transform(x_test)
    # print(selection_x_train)

    selection_model =XGBRegressor()
    selection_model.fit(selection_x_train,y_train)
    
    y_pre = selection_model.predict(selection_x_test)
    select_r2 = r2_score(y_pre,y_test)
    
    print("idx")
    print(idx)
    print("thresh")
    print(thresh)
    print("select_r2")
    print(select_r2)
    # print("y_pre")
    # print(y_pre)