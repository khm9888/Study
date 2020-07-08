import tensorflow as tf
import numpy as np



dataset = np.loadtxt("data\csv\data-01-test-score.csv",delimiter=",",dtype=np.float32)

x_data=dataset[:,:-1]
y_data=dataset[:,[-1]]


x=tf.placeholder(tf.float32,shape=[None,3])
y=tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([3,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=3e-2)

train=optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val,hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={x:x_data,y:y_data})
        # print(type(hy_val))
        # print(hy_val.shape[0])
        if step%10==1:
            print(f"step:{step},cost_val:{cost_val}")
            for i in range(hy_val.shape[0]):
                print(f"hy_val_{i}:{hy_val[i]}  y_data_{i}:{y_data[i]}")
