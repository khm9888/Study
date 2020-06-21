import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score,mean_absolute_error as mae
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor,plot_importance

for size in np.arange(0.97,1,0.01):
    #0~1 데이터 전처리, 데이터 입력

    train = pd.read_csv("./dacon/comp1_bio/train.csv",index_col=0,header=0,encoding="cp949")
    test = pd.read_csv("./dacon/comp1_bio/test.csv",index_col=0,header=0,encoding="cp949")
    submission = pd.read_csv("./dacon/comp1_bio/csv/sample_submission.csv",index_col=0,header=0,encoding="cp949")

    # for i in range(len()):
        
        
    # print(train.shape)#(10000, 75) train,test
    # print(test.shape)#(10000, 71) x_predict
    # print(submission.shape)#(10000, 4) y_predict


    # print(train.isnull().sum())
                                        
    # train = train.interpolate()#보간법 // 선형보간

    # print(dir(train))
    # print(train.isnull().sum())


    test = test.interpolate() #보간법 // 선형보간

    # print(type(train))

    # train = train.fillna(method="pad",axis=1)
    test = test.fillna(method="pad",axis=1)


    # print(test.info())

    test=test.values


    # print(train.shape)

    x = train.values[:,:-4]
    y = train.values[:,-4:]



    # print(x.shape)
    # print(y.shape)

    x_train, x_test, y_train, y_test = tts(x,y,train_size=size)

    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    #2.모델 구성-0
    # 33
    # 47
    # 49
    # 5

    xgb_0 = XGBRegressor()
    xgb_1 = XGBRegressor()
    xgb_2 = XGBRegressor()
    xgb_3 = XGBRegressor()

    y_train_0=y_train[:,0]
    y_train_1=y_train[:,1]
    y_train_2=y_train[:,2]
    y_train_3=y_train[:,3]

    y_test_0 = y_test[:,0]
    y_test_1 = y_test[:,1]
    y_test_2 = y_test[:,2]
    y_test_3 = y_test[:,3]
    
    parameters = {
    # "n_estimators" : np.arange(100,301,100),
    # "learning_rate" : np.arange(0.01,0.03,0.01),
    # "colsample_bytree":np.arange(0.6,1,0.1),
    # "colsample_bylevel":np.arange(0.6,1,0.1),
    # "max_depth" : [4,5,6]
    }

    xgb_0.fit(x_train,y_train_0)
    xgb_1.fit(x_train,y_train_1)
    xgb_2.fit(x_train,y_train_2)
    xgb_3.fit(x_train,y_train_3)

    thresholds_0 = np.sort(xgb_0.feature_importances_)
    thresholds_1 = np.sort(xgb_1.feature_importances_)
    thresholds_2 = np.sort(xgb_2.feature_importances_)
    thresholds_3 = np.sort(xgb_3.feature_importances_)

    selection_0=SelectFromModel(xgb_0,threshold=thresholds_0[33],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    selection_1=SelectFromModel(xgb_1,threshold=thresholds_1[47],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    selection_2=SelectFromModel(xgb_2,threshold=thresholds_2[49],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    selection_3=SelectFromModel(xgb_3,threshold=thresholds_3[5],prefit=True)#median +  GridSearch까지 할 것.데이콘 적용해라.
    
    # xgb_0=GridSearchCV(xgb_0,parameters,cv=2,n_jobs=-1)
    # xgb_1=GridSearchCV(xgb_0,parameters,cv=2,n_jobs=-1)
    # xgb_2=GridSearchCV(xgb_0,parameters,cv=2,n_jobs=-1)
    # xgb_3=GridSearchCV(xgb_0,parameters,cv=2,n_jobs=-1)
    
    xgb_0.fit(x_train,y_train_0)
    xgb_1.fit(x_train,y_train_1)
    xgb_2.fit(x_train,y_train_2)
    xgb_3.fit(x_train,y_train_3)
    

    x_train_0 = selection_0.transform(x_train)
    x_train_1 = selection_1.transform(x_train)
    x_train_2 = selection_2.transform(x_train)
    x_train_3 = selection_3.transform(x_train)

    x_test_0 = selection_0.transform(x_test)
    x_test_1 = selection_1.transform(x_test)
    x_test_2 = selection_2.transform(x_test)
    x_test_3 = selection_3.transform(x_test)

    test_0 = selection_0.transform(test)
    test_1 = selection_1.transform(test)
    test_2 = selection_2.transform(test)
    test_3 = selection_3.transform(test)
            
    xgb_0.fit(x_train_0,y_train_0)
    xgb_1.fit(x_train_1,y_train_1)
    xgb_2.fit(x_train_2,y_train_2)
    xgb_3.fit(x_train_3,y_train_3)

    y_pre_0=xgb_0.predict(x_test_0)
    y_pre_1=xgb_1.predict(x_test_1)
    y_pre_2=xgb_2.predict(x_test_2)
    y_pre_3=xgb_3.predict(x_test_3)

    r2_0= xgb_0.score(x_test_0,y_test_0)
    r2_1= xgb_1.score(x_test_1,y_test_1)
    r2_2= xgb_2.score(x_test_2,y_test_2)
    r2_3= xgb_3.score(x_test_3,y_test_3)

    mae_0= mae(y_test_0,y_pre_0)
    mae_1= mae(y_test_1,y_pre_1)
    mae_2= mae(y_test_2,y_pre_2)
    mae_3= mae(y_test_3,y_pre_3)

    # m=list(mae_0,mae_1,mae_2,mae_3)
    mae_result = (mae_0+mae_1+mae_2+mae_3)/4
    r2_result = (r2_0+r2_1+r2_2+r2_3)/4

    print(__file__)
    print(size)
    
    # print(f"mae_0 : {mae_0}")
    # print(f"mae_1 : {mae_1}")
    # print(f"mae_2 : {mae_2}")
    # print(f"mae_3 : {mae_3}")

    # print(f"mae_result : {mae_result}")

    # print(f"r2_0 : {r2_0}")
    # print(f"r2_1 : {r2_1}")
    # print(f"r2_2 : {r2_2}")
    # print(f"r2_3 : {r2_3}")

    print(f"r2_result : {r2_result}")
    #################################################################################
    # 33
    # 47
    # 49
    # 5


    # test_0 = selection_0.transform(test)
    # test_1 = selection_1.transform(test)
    # test_2 = selection_2.transform(test)
    # test_3 = selection_3.transform(test)

    # print("*"*30,"truly","*"*30)

    y_pre_0=xgb_0.predict(test_0)
    y_pre_1=xgb_1.predict(test_1)
    y_pre_2=xgb_2.predict(test_2)
    y_pre_3=xgb_3.predict(test_3)

    y_pre=[y_pre_0,y_pre_1,y_pre_2,y_pre_3]
    y_pre=np.array(y_pre)
    y_pre=y_pre.transpose()

    numbering=1
    submission = pd.DataFrame(y_pre,np.arange(10000,20000))
    submission.to_csv(f"dacon\comp1_bio\csv\sample_submission_{__file__[-9:-3]}_{numbering}_{size}.csv", index = True, header=['hhb','hbo2','ca','na'],index_label='id')

    submit = pd.DataFrame({})


# 0.8999999999999999
# mae_0 : 1.0636574529266358
# mae_1 : 0.6831143217277527
# mae_2 : 1.9041240909957886
# mae_3 : 1.4003469899892806
# mae_result : 1.2628107139098645
# r2_0 : 0.76769667823695
# r2_1 : 0.2731110717715566
# r2_2 : 0.3324031966687432
# r2_3 : 0.13255064558890528
# r2_result : 0.37644039806653873




# 0.91
# mae_0 : 1.0601455481039153
# mae_1 : 0.6677339043935141
# mae_2 : 1.8885800467703078
# mae_3 : 1.3891805051498942
# mae_result : 1.2514100011044078
# r2_0 : 0.7728370967152299
# r2_1 : 0.2991981063555704
# r2_2 : 0.3291873922812134
# r2_3 : 0.133194619350556
# r2_result : 0.3836043036756424
# None
# __main__
# d:\Study\dacon\comp1_bio\xgb04_result.py
# 0.92
# mae_0 : 1.0581199690461158
# mae_1 : 0.6611929723620414
# mae_2 : 1.887303155016899
# mae_3 : 1.4023980704993009
# mae_result : 1.2522535417310894
# r2_0 : 0.7713815465562448
# r2_1 : 0.3252192858406978
# r2_2 : 0.31432026829841697
# r2_3 : 0.10865231875842796
# r2_result : 0.3798933548634469
# __main__
# d:\Study\dacon\comp1_bio\xgb04_result.py
# 0.93
# mae_0 : 1.0491020107269287
# mae_1 : 0.6484761710439408
# mae_2 : 1.8707804099491665
# mae_3 : 1.3875017716850553
# mae_result : 1.2389650908512728
# r2_0 : 0.7595924134834671
# r2_1 : 0.3560778109730862
# r2_2 : 0.3603950109373918
# r2_3 : 0.11460368858446823
# r2_result : 0.3976672309946033

# d:\Study\dacon\comp1_bio\xgb04_result.py
# 0.9400000000000001
# mae_0 : 1.0435795399049916
# mae_1 : 0.6945572985967
# mae_2 : 1.905450824546814
# mae_3 : 1.4428472489674888
# mae_result : 1.2716087280039985
# r2_0 : 0.7640536469671666
# r2_1 : 0.31440232478601626
# r2_2 : 0.327718100798
# r2_3 : 0.06729477373789017
# r2_result : 0.36836721157226826

# __main__
# d:\Study\dacon\comp1_bio\xgb04_result.py
# 0.9500000000000001
# mae_0 : 1.0179038446474076
# mae_1 : 0.6595030723381041
# mae_2 : 1.8666208945560454
# mae_3 : 1.43758286601305
# mae_result : 1.245402669388652
# r2_0 : 0.7848654591085766
# r2_1 : 0.3565595243118468
# r2_2 : 0.33279079725427485
# r2_3 : 0.10286590434599052
# r2_result : 0.3942704212551722

# __main__
# d:\Study\dacon\comp1_bio\xgb04_result.py
# 0.9600000000000001
# mae_0 : 1.0635845565259456
# mae_1 : 0.6759656991600991
# mae_2 : 1.8676578086853026
# mae_3 : 1.3644282807350157
# mae_result : 1.2429090862765908
# r2_0 : 0.7598875006118541
# r2_1 : 0.28699493192435277
# r2_2 : 0.3301668077582822
# r2_3 : 0.14463472201425942
# r2_result : 0.38042099057718715

# __main__
# d:\Study\dacon\comp1_bio\xgb04_result.py
# 0.9700000000000001
# mae_0 : 0.9623052700360616
# mae_1 : 0.6494294956843057
# mae_2 : 1.848511576271057
# mae_3 : 1.3107077311674755
# mae_result : 1.192738518289725
# r2_0 : 0.7935301963131351
# r2_1 : 0.296615404475994
# r2_2 : 0.32333209134885865
# r2_3 : 0.21301394923520423
# r2_result : 0.406622910343298

