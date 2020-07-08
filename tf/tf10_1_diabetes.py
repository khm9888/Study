import tensorflow as tf
import numpy as np
from sklearn.datasets import load_diabetes


# print(type(load_diabetes()))
datasets=load_diabetes()
# print(datasets.keys())
x_data=datasets.data
y_data=datasets.target

print(x_data.shape)
print(y_data.shape)

y_data = y_data.reshape(y_data.shape[0],1)


x=tf.placeholder(tf.float32,shape=[None,10])
y=tf.placeholder(tf.float32,shape=[None,1])
w = tf.Variable(tf.random_normal([10,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
train = optimizer.minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val,hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={x:x_data,y:y_data})
        # print(type(hy_val))
        # print(hy_val.shape[0])
        if step%100==1:
            print(f"step:{step},cost_val:{cost_val}")
            for i in range(20):
                print(f"hy_val_{i}:{hy_val[i]}  y_data_{i}:{y_data[i]}")
'''
step:1901,cost_val:6380.91357421875
'''