import pandas as pd 
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')


#1. 데이터
iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

# print(x)
# print(y)

warnings.filterwarnings('ignore')


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

kfol = KFold(n_splits=5, shuffle=True)   # 5개씩 나눠서 20프로씩 나누겠다 20프로 검증 80프로 테스트 (각 데이터를 검증 할수 있음)
print()
for i in kfol.split(x):
    print(i )
print()

warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')  # 모든 클래스파이어값들이 들어있음

for (name, algorithm) in allAlgorithms:   # 올알고리즘에서 반환하는 값이 (네임, 알고리즘)
    model = algorithm()

    scores = cross_val_score(model, x, y, cv=kfol) # x,y통으로 들어간걸 5개로 짤라줘서 스코어를 내주겠따 (모델과, 엑스값, 와이값, 몇등분)
    print(name, '의 정답률 = ')
    print(scores)