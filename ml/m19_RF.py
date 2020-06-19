from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.2, random_state=42)

# model = DecisionTreeClassifier(max_depth=4) # max_depth = tree 그림을 그렸을때 깊이
model = RandomForestClassifier() 

#max_features :기본값 쓰기
#n_estimators : 클수록 좋다 단점 메모리 많이 차지 기본값 100
#n_jobs=-1    : 병렬처리


model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_)


# 유방암 30개 칼럼 중 가장 필요한 특징들 퍼센트 나오는거 (pca랑 비슷함)


import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()