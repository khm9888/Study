import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils.testing import all_estimators
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

all_Algorithms = all_estimators(type_filter="regressor")

for name,algorithms in all_Algorithms:
    model = algorithms()
    model.fit(x_train,y_train)
    y_pre=model.predict(x_test)
    # print(f"{name}의 정답률 = {accuracy_score(y_test,y_pre)}")
    
    print(f"{name}의 정답률 = {model.score(x_test,y_test)}")
    # print(f"{name}의 정답률 = {model.score(x_test,y_test)}")
    print(y_pre)
    
import sklearn
print(sklearn.__version__)