
def func(xgb):
    import numpy as np

    thresholds = np.sort(xgb.feature_importances_)

    idx_max = -1
    max = r2

    for idx,thresh in enumerate(thresholds):
        #데이터 전처리
        selection = SelectFromModel(xgb,threshold=thresh,prefit=True)
        #1)데이터입력
        selection_x_train = selection.transform(x_train)
        selection_x_test = selection.transform(x_test)
        #2)모델구성
        selection_model = XGBRegressor(n_estimators=100,learning_rate = 0.1,n_jobs=-1)
        
        #3)훈련
        selection_model.fit(selection_x_train,y_train,verbose=False, eval_metric=["logloss","rmse"],eval_set=[(selection_x_train, y_train), (selection_x_test,y_test)],early_stopping_rounds=20)
        
        #4)평가 및 예측
        y_pre = selection_model.predict(selection_x_test)
        r2 = r2_score(y_test,y_pre)
        print("idx")
        print(idx)
        print("r2")
        print(r2)
        if max<=r2:
            max=r2
            idx_max=idx

    print("idx_max")
    print(idx_max)
    print("max")
    print(max)