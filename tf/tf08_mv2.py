import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.random.randint(17,100,15)
x_data=x_data.reshape(5,3)
y_data = np.random.randint(150,200,5)
y_data=y_data.reshape(5,1)

print(x_data)
print(y_data)


x=tf.placeholder(tf.float32,shape=[None,3])
y=tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([3,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)

train=optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val,hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={x:x_data,y:y_data})
        if step%10==1:
            print(f"step:{step},cost_val:{cost_val}")
            print(f"hy_val:{hy_val}")
            print(f"y_data:{y_data}")
