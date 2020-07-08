# 이진분류

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_breast_cancer

diabetes = load_breast_cancer()
x_data = diabetes['data']
y_data = diabetes['target']

print("x_data.shape :", x_data.shape) # (569, 30)
print("y_data.shape :", y_data.shape) # (569,)


y_data = y_data.reshape(y_data.shape[0], 1)
# print("y_data.shape :", y_data.shape)

# tensorflow 에서 one_hot
# y_data = tf.one_hot(y_data, depth = 2, dtype = tf.float32)
# print("y_data.shape :", y_data.shape) # (569, 2)
# shape 2로 맞춰주고 돌렸는데 안 됨, 다른 게 필요한 듯

x = tf.placeholder(tf.float32, shape=[None, 30])          
y = tf.placeholder(tf.float32, shape=[None, 1])          


print(x) # Tensor("Placeholder:0", shape=(?, 30), dtype=float32)
print(y) # Tensor("Placeholder_1:0", shape=(?, 2), dtype=float32)


w = tf.Variable(tf.zeros([30, 1]), name = 'weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')


hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # wx+b

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000002)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) 

    for step in range(5001) :
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})

        if step % 10 == 0 :
            print(step, "cost :", cost_val)

        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_data, y:y_data})
        # print("\n Hypothesis :", h, "\n Correct (y) :", c, "\n Accuracy :", a)
        # print("-------------------------")
        # print("Hypothesis")
        # print(h)
        # print("-------------------------")
        # print("Correct(y)")
        # print(c)
        # print("-------------------------")
        print("Accuracy")
        print(a)


# (learning_rate = 0.0000002)
# Accuracy
# 0.90685415