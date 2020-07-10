# layer = tf.nn.softmax(tf.matmul(x,w)+b)
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# 탠서플로의 원핫 인코딩
# aaa = tf.one_hot(y, ???)
seed = 0
tf.set_random_seed(seed)

(x_data, y_data), (x_test, y_test) = mnist.load_data()

print(x_data.shape)
print(y_data.shape)


sess = tf.Session()
y_data = tf.one_hot(y_data,depth=10,on_value=1,off_value=0).eval(session=sess)
y_test = tf.one_hot(y_test,depth=10,on_value=1,off_value=0).eval(session=sess)
sess.close()
print(y_data.shape)

x_data = x_data.reshape(-1,x_data.shape[1]*x_data.shape[2])
print(x_data.shape)
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])
print(x_test.shape)

x_col_num = x_data.shape[1] 
y_col_num = y_data.shape[1] 

print(x_col_num)
print(y_col_num)

x = tf.placeholder(tf.float32, shape=[None, x_col_num])
y = tf.placeholder(tf.float32, shape=[None, y_col_num])

w1 = tf.Variable(tf.zeros([x_col_num, 512]), name = 'weight1') # 다음 레이어에 100개를 전달? 노드의 갯수와 동일하다고 봐도 무방?
b1 = tf.Variable(tf.zeros([512]), name = 'bias1') # 자연스럽게 100을따라감 ?? 왜?
layer1 = tf.matmul(x, w1) + b1

w2 = tf.Variable(tf.random_normal([512, 256]), name='weight2')
b2 = tf.Variable(tf.random_normal([256]),name = 'bias2')
layer2 = tf.matmul(layer1, w2) + b2

w3 = tf.Variable(tf.random_normal([256, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]),name = 'bias3')
h = tf.nn.softmax(tf.matmul(layer2, w3) + b3)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h),axis=1)) # loss ... 계산 방법 ...

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(loss) # 어떻게 쓰는지 어떻게 계산하는지 지금은 일단 쓰고 시간이 많을때 꼭!!! 공부하라 경사하강법

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        _, _, cost_val = sess.run([h, opt, loss], feed_dict={ x: x_data, y: y_data})

        # if i % 200 == 0 :
        print( i , cost_val)
    
    pred = sess.run(h, feed_dict={x:x_test}) # keras model.predict(x_test_data)
    pred = sess.run(tf.argmax(pred, 1)) # tf.argmax(a, 1) 안에 값들중에 가장 큰 값의 인덱스를 표시하라
    # pred = pred.reshape(-1,1)
    print(pred)

    y_test = sess.run(tf.argmax(y_test,1))
    print(y_test)

    acc = tf.reduce_mean(tf.compat.v1.cast(tf.equal(pred, y_test),tf.float32))
    acc = sess.run(acc)
    print(acc)