import pandas as pd
from sklearn.model_selection import train_test_split as tts,cross_val_score,KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv("./data/csv/iris.csv",header=0)

x=iris.iloc[:,0:4]
y=iris.iloc[:,4]

print(y)

x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)

all_Algorithms = all_estimators(type_filter="classifier")
kfol = KFold(n_splits=5, shuffle=True)
good=[]
bad=[]
most=[]
value=0
# print(all_Algorithms)
for name,algorithms in all_Algorithms:
    try:
        model = algorithms()
        model.fit(x_train,y_train)
        y_pre=model.predict(x_test)
        acc = cross_val_score(model,x_test,y_test,cv=kfol)#[0.86666667 0.83333333 0.93333333 1.         0.93333333]
        print(name)
        print(acc)
        if max(acc)>=0.9:
            good.append(f"{name}의 정답률 = {max(acc)}")
        # print(f"{name}의 정답률 = {acc}")
        else:
            bad.append(f"{name}의 정답률 = {max(acc)}")
        if value<=acc and acc==1:
            most.append(name)
            value=max(acc)
    except:
        pass

print("------------most-----------")
print(most)
print(value)
print("------------good-----------")
for al in good:
    print(al)
    
print("------------bad-----------")
for al in bad:
    print(al)