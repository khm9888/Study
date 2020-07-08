import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.random.randint(1,6,12)
x_data=x_data.reshape(6,2)
y_data = np.array([0,0,0,1,1,1])
y_data=y_data.reshape(6,1)

print(x_data)
print(y_data)


x=tf.placeholder(tf.float32,shape=[None,2])
y=tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([2,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=9e-2)#과연 몇으로 해야하는가?

train=optimizer.minimize(cost)

predicted=tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(501):
        cost_val,hy_val,_,acc_val = sess.run([cost,hypothesis,train,accuracy],feed_dict={x:x_data,y:y_data})
        # print(type(hy_val))
        # print(hy_val.shape[0])
        if step%10==1:
            print(f"step:{step},cost_val:{cost_val},acc_val:{acc_val}")
            for i in range(hy_val.shape[0]):
                print(f"hy_val_{i}:{hy_val[i]}  y_data_{i}:{y_data[i]}")

               
