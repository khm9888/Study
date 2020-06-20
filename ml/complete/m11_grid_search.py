import pandas as pd
from sklearn.model_selection import train_test_split as tts, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC



iris = pd.read_csv("./data/csv/iris.csv",index_col=None,header=0)

# iris.info()
# print(iris.head())

x=iris.iloc[:,0:4]
y=iris.iloc[:,4]
# print(x)
# x= iris[]
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)

paraemeters = [
    {"C":[1,10,100,1000],"kernel":["linear"]},
    {"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[0.001,0.0001]},
    {"C":[1,10,100,1000],"kernel":["sigmoid"],"gamma":[0.001,0.0001]}
]

kfold = KFold(n_splits=5,shuffle=True)



model = GridSearchCV(SVC(),paraemeters,cv=kfold,n_jobs=-1)


model.fit(x_train,y_train)

y_pre = model.predict(x_test)

print(f"최적의 매개변수 = {model.best_estimator_}")
print(f"최종 정답률 = {accuracy_score(y_test,y_pre)}")

