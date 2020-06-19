from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=4) # max_depth = tree 그림을 그렸을때 깊이

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_)


# 유방암 30개 칼럼 중 가장 필요한 특징들 퍼센트 나오는거 (pca랑 비슷함)
'''
acc :  0.8903508771929824
[0.02518162 0.         0.         0.         0.036894   0.
 0.         0.81604753 0.01888621 0.         0.         0.01718717
 0.         0.         0.         0.         0.         0.
 0.         0.03498776 0.         0.0508157  0.         0.
 0.         0.         0.         0.         0.         0.        ]
'''
