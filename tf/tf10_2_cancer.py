import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer


# print(type(load_diabetes()))
datasets=load_breast_cancer()
# print(datasets.keys())
x_data=datasets.data
y_data=datasets.target

print(x_data.shape)
print(y_data.shape)

y_data = y_data.reshape(y_data.shape[0],1)


x=tf.placeholder(tf.float32,shape=[None,30])
y=tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.zeros ([30,1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000002)
train = optimizer.minimize(cost)

predicted=tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})

        if step % 10 == 0 :
            print(step, "cost :", cost_val)

        hy_val, c, acc_val = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_data, y:y_data})
        # print(type(hy_val))
        # print(hy_val.shape[0])
        if step%10==1:
            print(f"step:{step},cost_val:{cost_val},acc_val:{acc_val}")
            for i in range(10):
                print(f"hy_val_{i}:{hy_val[i]}  y_data_{i}:{y_data[i]}")
'''
step:4991,cost_val:0.4493256211280823,acc_val:0.9068541526794434
hy_val_0:[0.00430748]  y_data_0:[0]
hy_val_1:[0.08232763]  y_data_1:[0]
hy_val_2:[0.16423652]  y_data_2:[0]
hy_val_3:[0.49749908]  y_data_3:[0]
hy_val_4:[0.5146868]  y_data_4:[0]
hy_val_5:[0.36640382]  y_data_5:[0]
hy_val_6:[0.11100736]  y_data_6:[0]
hy_val_7:[0.29854593]  y_data_7:[0]
hy_val_8:[0.47014368]  y_data_8:[0]
hy_val_9:[0.4267061]  y_data_9:[0]
5000 cost : 0.44914776
'''