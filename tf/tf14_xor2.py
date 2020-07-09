import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([[0],[1],[1],[0]])

print(x_data)
print(y_data)


x=tf.placeholder(tf.float32,shape=[None,2])
y=tf.placeholder(tf.float32,shape=[None,1])

w1 = tf.Variable(tf.random_normal([2,100]),name="weight1")
b1 = tf.Variable(tf.random_normal([100]),name="bias1")
layer1 = tf.sigmoid(tf.matmul(x,w1)+b1)
#model.add(Dense(100,input_shape=(2,)))

w2 = tf.Variable(tf.random_normal([100,50]),name="weight1")
b2 = tf.Variable(tf.random_normal([50]),name="bias1")
layer2 = tf.sigmoid(tf.matmul(layer1,w2)+b2)
#model.add(Dense(50))

w3 = tf.Variable(tf.random_normal([50,1]),name="weight1")
b3 = tf.Variable(tf.random_normal([1]),name="bias1")
hypothesis = tf.sigmoid(tf.matmul(layer2,w3)+b3)
#model.add(Dense(1)

#activation -sigmoid- relu


cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=9e-2)#과연 몇으로 해야하는가?

train=optimizer.minimize(cost)

predicted=tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10000):
        cost_val,hy_val,_,acc_val = sess.run([cost,hypothesis,train,accuracy],feed_dict={x:x_data,y:y_data})
        # print(type(hy_val))
        # print(hy_val.shape[0])
        if step%100==1:
            print(f"step:{step},cost_val:{cost_val},acc_val:{acc_val}")
            for i in range(hy_val.shape[0]):
                print(f"hy_val_{i}:{hy_val[i]}  y_data_{i}:{y_data[i]}")

               