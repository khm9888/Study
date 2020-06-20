import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings


iris = pd.read_csv("./data/csv/iris.csv",index_col=None,header=0)

warnings.filterwarnings('ignore')

x=iris.iloc[:,0:4]
y=iris.iloc[:,4]

x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)


all_Algorithms = all_estimators(type_filter="classifier")

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
        if accuracy_score(y_test,y_pre)>=0.9:
            good.append(f"{name}의 정답률 = {accuracy_score(y_test,y_pre)}")
        # print(f"{name}의 정답률 = {model.score(x_test,y_test)}")
        else:
            bad.append(f"{name}의 정답률 = {accuracy_score(y_test,y_pre)}")
        if value<=model.score(x_test,y_test) and acc==1:
        
            most.append(name)
            value=model.score(x_test,y_test)
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

'''
------------most-----------
['AdaBoostClassifier', 'BaggingClassifier', 'DecisionTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB', 'GaussianProcessClassifier', 'GradientBoostingClassifier', 'KNeighborsClassifier', 'LabelPropagation', 'LabelSpreading', 'LinearDiscriminantAnalysis', 'MLPClassifier', 'NuSVC', 'QuadraticDiscriminantAnalysis', 'RandomForestClassifier', 'SVC']
1.0
------------good-----------
AdaBoostClassifier의 정답률 = 1.0
BaggingClassifier의 정답률 = 1.0
CalibratedClassifierCV의 정답률 = 0.9333333333333333
DecisionTreeClassifier의 정답률 = 1.0
ExtraTreeClassifier의 정답률 = 0.9333333333333333
ExtraTreesClassifier의 정답률 = 1.0
GaussianNB의 정답률 = 1.0
GaussianProcessClassifier의 정답률 = 1.0
GradientBoostingClassifier의 정답률 = 1.0
KNeighborsClassifier의 정답률 = 1.0
LabelPropagation의 정답률 = 1.0
LabelSpreading의 정답률 = 1.0
LinearDiscriminantAnalysis의 정답률 = 1.0
LinearSVC의 정답률 = 0.9666666666666667
LogisticRegression의 정답률 = 0.9333333333333333
LogisticRegressionCV의 정답률 = 0.9666666666666667
MLPClassifier의 정답률 = 1.0
MultinomialNB의 정답률 = 0.9666666666666667
NearestCentroid의 정답률 = 0.9333333333333333
NuSVC의 정답률 = 1.0
QuadraticDiscriminantAnalysis의 정답률 = 1.0
RadiusNeighborsClassifier의 정답률 = 0.9666666666666667
RandomForestClassifier의 정답률 = 1.0
SVC의 정답률 = 1.0
------------bad-----------
BernoulliNB의 정답률 = 0.3
ComplementNB의 정답률 = 0.6666666666666666
PassiveAggressiveClassifier의 정답률 = 0.7
Perceptron의 정답률 = 0.6666666666666666
RidgeClassifier의 정답률 = 0.8333333333333334
RidgeClassifierCV의 정답률 = 0.8333333333333334
SGDClassifier의 정답률 = 0.6666666666666666
'''