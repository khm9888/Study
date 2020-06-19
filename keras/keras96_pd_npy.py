import matplotlib.pyplot        as plt
import numpy                    as np
import matplotlib.pyplot        as plt
from   sklearn.preprocessing    import StandardScaler, MinMaxScaler
from   sklearn.decomposition    import PCA
from   sklearn.model_selection  import train_test_split
from   keras.utils              import np_utils
from   keras.models             import Sequential
from   keras.layers             import Dense
from   keras.callbacks          import EarlyStopping, ModelCheckpoint
# 데이터 로드
iris_data = np.load('./data/iris_data.npy')

print(iris_data)
print(iris_data.shape)


# 슬라이싱
x = iris_data[:,:4]
y = iris_data[:,-1]

# print(x)
# print(y)

# print(x.shape) # (150,4)
# print(y.shape) # (150,)

# 스칼라 전처리
scaler = MinMaxScaler()
scaler.fit(x)   
x = scaler.transform(x)



# 트레인테스트 스플릿
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state = 66, shuffle=True, train_size=0.8)
# print(x_train.shape) # (120,4)
# print(y_train.shape) # (120,)
# print(x_test.shape) # (30,4)
# print(y_test.shape) # (30,)


# 원핫 인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(y_train.shape) # (120,3)
# print(y_test.shape) # (30,3)



print(x_train)
print(x_test)

# 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=(4)))
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 컴파일, 핏
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
ck = ModelCheckpoint(filepath = modelpath, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)
es = EarlyStopping(monitor='acc', patience=10, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, validation_split=0.5, epochs=33, batch_size=16, verbose=1, callbacks=[es,ck])

loss, acc = model.evaluate(x_test, y_test)

print('loss : ', loss)
print('acc : ',acc)

y_predict = model.predict(x_test)

# print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_predict)
print(y_test)
'''
print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_predict[0:20]}")
'''
# print('acc : ', acc)
# print('val_acc : ', val_acc)
# print('loss_acc : ', loss,acc)
'''
# 그래프 표현
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

plt.figure(figsize=(10, 6)) #가로세로 길이 그래프설정

plt.subplot(2, 1, 1) #두개의 그림을 그림  (2행 1열에 첫번째 그림을 그리겠다)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')  # 에코가x라서 x값은 안넣었는데 만약 들어가면 이런식으로 plt.plot(x, hist.history['loss'])
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') #label 레전드안에 안써주고 라벨을 따로 붙혀줄수있다
plt.grid() # 모눈종이처럼 가로세로 줄이 그어짐
plt.title('loss')        # 제목
plt.ylabel('loss')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(loc = 'upper right')  #plot 순서에따라 맞춰서 기입   #upperright는 legend 위치 명시


plt.subplot(2, 1, 2) #두개의 그림을 그림  (2행 1열에 첫번째 그림을 그리겠다)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid() # 모눈종이처럼 가로세로 줄이 그어짐
plt.title('acc')        # 제목
plt.ylabel('acc')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(['acc', 'val_acc'])

plt.show()
'''


'''
loss :  0.2810623049736023
acc :  0.9666666388511658
y_test[0:20]:[2 2 2 1 2 2 1 1 1 3 3 3 1 3 3 1 2 2 3 3]
y_pre[0:20]:[2 2 2 1 2 2 1 1 1 3 3 3 1 3 3 1 2 3 3 3]

레이어 추가
loss :  0.16412195563316345
acc :  0.9333333373069763
y_test[0:20]:[2 2 2 1 2 2 1 1 1 3 3 3 1 3 3 1 2 2 3 3]
y_pre[0:20]:[2 2 2 1 2 2 1 1 1 3 3 3 1 3 3 1 2 3 3 3]

레이어 추가
loss :  0.16842810809612274
acc :  0.9666666388511658
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2]
'''
