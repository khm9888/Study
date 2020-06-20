from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#1.데이터 입력
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#2.모델

# model =SVC()
model =KNeighborsClassifier(n_neighbors=1)#통상적으로 4이상 별로 안 씀. 3이하로 사용

#3.훈련

model.fit(x_data,y_data)

#4. 평가, 예측

x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)

acc = accuracy_score(y_data,y_predict)

print(x_test,"의 예측결과",y_predict)
print("acc = ",acc)