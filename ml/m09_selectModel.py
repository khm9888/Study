import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.metrics import accuracy_score
import warnings




iris = pd.read_csv("./data/csv/iris.csv",index_col=None,header=0)

warnings.filterwarnings('ignore')

# iris.info()
print(iris.head())

x=iris.iloc[:,0:4]
y=iris.iloc[:,4]
# print(x)
# x= iris[]


x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8)

all_Algorithms = all_estimators(type_filter="classifier")

for name,algorithms in all_Algorithms:
    model = algorithms()
    model.fit(x_train,y_train)
    y_pre=model.predict(x_test)
    print(f"{name}의 정답률 = {accuracy_score(y_test,y_pre)}")
    print(f"{name}의 정답률 = {model.score(x_test,y_test)}")
    
import sklearn

print(sklearn.__version__)