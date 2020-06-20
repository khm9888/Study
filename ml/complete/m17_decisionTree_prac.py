from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.2, random_state=42)

model=DecisionTreeClassifier(max_depth=3)

model.fit(x_train,y_train)
acc = model.score(x_test,y_test)

print(model.feature_importances_)




