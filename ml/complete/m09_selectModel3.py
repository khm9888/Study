import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x)
print(y)

warnings.filterwarnings('ignore')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)


warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')  # 모든 클래스파이어값들이 들어있음

for (name, algorithm) in allAlgorithms:   # 올알고리즘에서 반환하는 값이 (네임, 알고리즘)
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, '의 정답률 = ', accuracy_score(y_test, y_pred))