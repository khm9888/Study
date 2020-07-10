# 20-07-09_32
# deep learning _dnn mnist

import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
layer = tf.nn.relu(tf.matmul(layer, w) + b)
layer = tf.nn.elu(tf.matmul(layer, w) + b)
layer = tf.nn.selu(tf.matmul(layer, w) + b)
layer = tf.nn.dropout(layer, keep_prob=0.3)
'''

# 1. data
tf.set_random_seed(777)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)  # (60000,) (10000,)

def min_max_scaler(dataset):
    numerator = dataset - np.min(dataset, 0)
    denominator = np.max(dataset, 0) - np.min(dataset, 0)
    return numerator / (denominator + 1e-7)

# 1-0. minmaxscaler
x_train = min_max_scaler(x_train)
x_test = min_max_scaler(x_test)
# print(x_train)
# print(x_test)

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
x_shape = (28 * 28)

# 1-1. one hot encoding & to numpy
sess = tf.Session()
y_train = sess.run(tf.one_hot(y_train, 10))
y_test = sess.run(tf.one_hot(y_test, 10))
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

x = tf.placeholder('float', shape=[None, x_shape])
y = tf.placeholder('float', shape=[None, 10])

w1 = tf.Variable(tf.zeros([x_shape, 10]), name='weight1')
b1 = tf.Variable(tf.zeros([10]), name='bias1')
# layer1 = tf.matmul(x, w1) + b1
hypothesis = tf.nn.softmax(tf.matmul(x, w1) + b1)


# w2 = tf.Variable(tf.zeros([100, 10]), name='weight2')        
# b2 = tf.Variable(tf.zeros([10]), name='bias2')
# layer2 = tf.nn.selu(tf.matmul(layer1, w2) + b2)

# w3 = tf.Variable(tf.zeros([100, 10]), name='weight3')        
# b3 = tf.Variable(tf.zeros([10]), name='bias3')
# hypothesis = tf.nn.softmax(tf.matmul(layer1, w3) + b3)

# 3. categorical cross entropy cost(loss)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# 4. gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.9).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 5. launch(model.fit)
train = {x: x_train, y: y_train}
test = {x: x_test, y: y_test}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(201):
        _, cost_val = sess.run([optimizer, cost], feed_dict=train)

        # if step % 500 == 0:
        print(f'{step} cost_v : {cost_val:.5f}')

    p, a = sess.run([hypothesis, accuracy], feed_dict=test)
    print('\n',p, '\n pred : ',sess.run(tf.argmax(p, 1)))
    print('acc : {:.2%}'.format(a))