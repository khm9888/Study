import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings




boston = pd.read_csv("./data/csv/boston_house_prices.csv",index_col=None,header=1)

print(boston.head())

warnings.filterwarnings('ignore')

# boston.info()
# print(boston.head())

x=boston.iloc[:,0:-1]
y=boston.iloc[:,-1]
# print(x)
# x= boston[]

x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)

all_Algorithms = all_estimators(type_filter='regressor')

good=[]
bad=[]
most=""
value=0

# print(all_Algorithms)

for name,algorithms in all_Algorithms:
    try:
        model = algorithms()
        model.fit(x_train,y_train)
        y_pre=model.predict(x_test)
        if model.score(x_test,y_test)>=0.8:
            good.append(f"{name}의 정답률 = {model.score(x_test,y_test)}")
        else:
            bad.append(f"{name}의 정답률 = {model.score(x_test,y_test)}")
        if value<model.score(x_test,y_test):
            most=name
            value=model.score(x_test,y_test)
    except:
        pass

# for (name, algorithm) in all_Algorithms:   # 올알고리즘에서 반환하는 값이 (네임, 알고리즘)
#     model = algorithm()
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     print(name, '의 정답률 = ', r2_score(y_test, y_pred))
print("------------most-----------")
print(most)
print(value)
print("------------good-----------")
for al in good:
    print(al)
    
print("------------bad-----------")
for al in bad:
    print(al)
'''
------------most-----------
GradientBoostingRegressor
0.8963339194607347
------------good-----------
ARDRegression의 정답률 = 0.817087512419024
AdaBoostRegressor의 정답률 = 0.8897105845513767
BaggingRegressor의 정답률 = 0.8941527646319789
BayesianRidge의 정답률 = 0.8121392720223428
CCA의 정답률 = 0.81734807596138
DecisionTreeRegressor의 정답률 = 0.8047569153855574
ExtraTreeRegressor의 정답률 = 0.8617136425540954
ExtraTreesRegressor의 정답률 = 0.8823780667720511
GradientBoostingRegressor의 정답률 = 0.8963339194607347
KernelRidge의 정답률 = 0.8319549107592183
Lars의 정답률 = 0.8087651873615083
LarsCV의 정답률 = 0.8087651873615083
LassoLarsCV의 정답률 = 0.8215747509149941
LassoLarsIC의 정답률 = 0.8239309232115037
LinearRegression의 정답률 = 0.8215747509149953
OrthogonalMatchingPursuitCV의 정답률 = 0.8201632677783497
PLSRegression의 정답률 = 0.8283035511452478
RandomForestRegressor의 정답률 = 0.8945906983869626
Ridge의 정답률 = 0.8243335239800413
RidgeCV의 정답률 = 0.8223294793079714
TheilSenRegressor의 정답률 = 0.8207099903754146
TransformedTargetRegressor의 정답률 = 0.8215747509149953
------------bad-----------
ElasticNet의 정답률 = 0.730680131020468
ElasticNetCV의 정답률 = 0.7043196878028775
GaussianProcessRegressor의 정답률 = -5.824595712911991
HuberRegressor의 정답률 = 0.7871771501281496
KNeighborsRegressor의 정답률 = 0.6192499235495212
Lasso의 정답률 = 0.7100240458286817
LassoCV의 정답률 = 0.7480296455339194
LassoLars의 정답률 = -0.009491198836089954
LinearSVR의 정답률 = 0.7439770532859962
MLPRegressor의 정답률 = 0.17170247506353886
NuSVR의 정답률 = -0.0322485585732355
OrthogonalMatchingPursuit의 정답률 = 0.5832352859291856
PLSCanonical의 정답률 = -1.6176204094310371
PassiveAggressiveRegressor의 정답률 = -0.10083562901227916
RANSACRegressor의 정답률 = 0.633508091424521
SGDRegressor의 정답률 = -3.6730580343564644e+26
SVR의 정답률 = -0.02166094765974913
'''