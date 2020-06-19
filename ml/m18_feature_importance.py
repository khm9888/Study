from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, 
                                                    cancer.target, train_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=4) # max_depth = tree 그림을 그렸을때 깊이

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_)
# [0.         0.         0.         0.         0.         0.
#  0.         0.81604753 0.         0.         0.         0.
#  0.         0.01888621 0.         0.         0.         0.
#  0.         0.03498776 0.         0.0508157  0.01718717 0.06207562
#  0.         0.         0.         0.         0.         0.        ]


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