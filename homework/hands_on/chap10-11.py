
# 공통
import numpy as np
import os

# 일관된 출력을 위해 유사난수 초기화
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# 맷플롯립 설정
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 한글출력
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

# 그림을 저장할 폴더
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


#################################################



import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import tensorflow as tf

iris = load_iris()
X = iris.data[:, (2, 3)]  # 꽃잎 길이, 꽃잎 너비
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron(max_iter=100, random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])

y_pred



#################################################

a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]
b = -per_clf.intercept_ / per_clf.coef_[0][1]

axes = [0, 5, 0, 2]

x0, x1 = np.meshgrid(
        np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
        np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]
y_predict = per_clf.predict(X_new)
zz = y_predict.reshape(x0.shape)




##################################################################

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs", label="Iris-Setosa 아님")
plt.plot(X[y==1, 0], X[y==1, 1], "yo", label="Iris-Setosa")

plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-", linewidth=3)
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#9898ff', '#fafab0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
plt.xlabel("꽃잎 길이", fontsize=14)
plt.ylabel("꽃잎 너비", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.axis(axes)

save_fig("perceptron_iris_plot")
plt.show()

##################################################################

def logit(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)
##################################################################
z = np.linspace(-5, 5, 200)

plt.figure(figsize=(11,4))

plt.subplot(121)
plt.plot(z, np.sign(z), "r-", linewidth=2, label="스텝")
plt.plot(z, logit(z), "g--", linewidth=2, label="로지스틱")
plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
plt.legend(loc="center right", fontsize=14)
plt.title("활성화 함수", fontsize=14)
plt.axis([-5, 5, -1.2, 1.2])

plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Step")
plt.plot(0, 0, "ro", markersize=5)
plt.plot(0, 0, "rx", markersize=10)
plt.plot(z, derivative(logit, z), "g--", linewidth=2, label="Logit")
plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
plt.title("도함수", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])

save_fig("activation_functions_plot")
plt.show()

##################################################################
def heaviside(z):
    return (z >= 0).astype(z.dtype)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def mlp_xor(x1, x2, activation=heaviside):
    return activation(-activation(x1 + x2 - 1.5) + activation(x1 + x2 - 0.5) - 0.5)


##################################################################
x1s = np.linspace(-0.2, 1.2, 100)
x2s = np.linspace(-0.2, 1.2, 100)
x1, x2 = np.meshgrid(x1s, x2s)

z1 = mlp_xor(x1, x2, activation=heaviside)
z2 = mlp_xor(x1, x2, activation=sigmoid)

plt.figure(figsize=(10,4))

plt.subplot(121)
plt.contourf(x1, x2, z1)
plt.plot([0, 1], [0, 1], "gs", markersize=20)
plt.plot([0, 1], [1, 0], "y^", markersize=20)
plt.title("활성화 함수: 헤비사이드", fontsize=14)
plt.grid(True)

plt.subplot(122)
plt.contourf(x1, x2, z2)
plt.plot([0, 1], [0, 1], "gs", markersize=20)
plt.plot([0, 1], [1, 0], "y^", markersize=20)
plt.title("활성화 함수: 시그모이드", fontsize=14)
plt.grid(True)

##################################################################
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

##################################################################

model  = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(1000,activation="relu"))
model.add(Dense(10,activation="softmax"))

##################################################################

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["acc"])
model.fit(X_train,y_train,epochs=5,batch_size=10,validation_split=0.3)

##################################################################
v=model.evaluate(X_test,y_test)
print(v)

'''

#1st -> sgd

import tensorflow as tf...
Train on 38500 samples, validate on 16500 samples
Epoch 1/5
38500/38500 [==============================] - 13s 344us/step - loss: 0.4636 - acc: 0.8790 - val_loss: 0.2904 - val_acc: 0.9176
Epoch 2/5
38500/38500 [==============================] - 13s 339us/step - loss: 0.2594 - acc: 0.9270 - val_loss: 0.2288 - val_acc: 0.9347
Epoch 3/5
38500/38500 [==============================] - 13s 339us/step - loss: 0.2074 - acc: 0.9416 - val_loss: 0.1977 - val_acc: 0.9446
Epoch 4/5
38500/38500 [==============================] - 13s 338us/step - loss: 0.1731 - acc: 0.9515 - val_loss: 0.1722 - val_acc: 0.9522
Epoch 5/5
38500/38500 [==============================] - 13s 338us/step - loss: 0.1485 - acc: 0.9582 - val_loss: 0.1567 - val_acc: 0.9561
10000/10000 [==============================] - 1s 58us/step
[0.1442225308768451, 0.9577999711036682]
'''

''''
#2nd

#adam

import tensorflow as tf...
Train on 38500 samples, validate on 16500 samples
Epoch 1/5
38500/38500 [==============================] - 16s 419us/step - loss: 0.2080 - acc: 0.9368 - val_loss: 0.1322 - val_acc: 0.9598
Epoch 2/5
38500/38500 [==============================] - 16s 414us/step - loss: 0.0914 - acc: 0.9711 - val_loss: 0.1007 - val_acc: 0.9710
Epoch 3/5
38500/38500 [==============================] - 16s 407us/step - loss: 0.0621 - acc: 0.9803 - val_loss: 0.1204 - val_acc: 0.9672
Epoch 4/5
38500/38500 [==============================] - 16s 404us/step - loss: 0.0442 - acc: 0.9864 - val_loss: 0.1340 - val_acc: 0.9679
Epoch 5/5
38500/38500 [==============================] - 16s 407us/step - loss: 0.0356 - acc: 0.9890 - val_loss: 0.1103 - val_acc: 0.9756
10000/10000 [==============================] - 1s 60us/step
[0.09563025432826652, 0.9765999913215637]
'''

##################################################################

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense
import numpy as np



housing = fetch_california_housing()
x_train,x_test,y_train,y_test = tts(housing.data,housing.target,train_size=0.8)
x_train,x_valid,y_train,y_valid = tts(x_train,y_train,train_size=7/8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_valid = scaler.transform(x_valid)



model  = Sequential([(Dense(50,activation="relu",input_shape=x_train.shape[1:]))
,Dense(1)])

model.compile(loss="mse",optimizer="sgd")
model.fit(x_train,y_train,epochs=5,batch_size=10,validation_split=0.3)

x_new = x_test[:3]

y_pred = model.predict(x_new)
print(y_pred)
##################################################################


##################################################################
