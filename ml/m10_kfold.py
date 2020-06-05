import pandas as pd
from sklearn.model_selection import train_test_split as tts,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.metrics import accuracy_score
import warnings




iris = pd.read_csv("./data/csv/iris.csv",index_col=None,header=0)

warnings.filterwarnings('ignore')

# iris.info()
# print(iris.head())

x=iris.iloc[:,0:4]
y=iris.iloc[:,4]
# print(x)
# x= iris[]

# 
k_fold =KFold(n_splits=5,shuffle=True)

all_Algorithms = all_estimators(type_filter="classifier")

for name,algorithms in all_Algorithms:
    model = algorithms()
    
    scores = cross_val_score(model,x,y,cv=k_fold)
    
    # y_pre=model.predict(x)
    print(f"{name}의 정답률 = {scores}")
    
import sklearn

print(sklearn.__version__)