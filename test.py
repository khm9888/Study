from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=66)

# XGBRFRegressor??????

model = XGBRegressor()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)

print(f"r2 : {score}")

'''feature engineering'''

thres_holds = np.sort(model.feature_importances_)
print(thres_holds)


# 반복문 안에다가 GridSearshCV를 엮어보기
for thresh in thres_holds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # 추가 파라미터 median

    selec_x_train = selection.transform(x_train)

    # print(f"selec_x_train.shape : {selec_x_train.shape}") # columns을 한개씩 줄이고 있다 

    selec_model = XGBRegressor()
    selec_model.fit(selec_x_train,y_train)

    selec_x_test = selection.transform(x_test)
    y_pred = selec_model.predict(selec_x_test)

    score = r2_score(y_test,y_pred)
    # print(score)
    # print(f"model.feature_importances_ : {model.feature_importances_}")

    print(f"Thresh={np.round(thresh,2)} \t n={selec_x_train.shape[1]} \t r2={np.round(score*100,2)}")

# 메일 제목 : 아무개 **등