import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.arange(1,33)
x_data=x_data.reshape(8,4)
y_data = np.array([0,0,1]*3+[0,1,0]*3+[1,0,0]*2)
# y_data = np.append(y_data,[1,1,0],axis=0)
y_data=y_data.reshape(-1,3)

print(x_data)
print(y_data)


x=tf.placeholder(tf.float32,shape=[None,4])
y=tf.placeholder(tf.float32,shape=[None,3])

w = tf.Variable(tf.random_normal([4,3]),name="weight")
b = tf.Variable(tf.random_normal([1,3]),name="bias")

hypothesis = tf.nn.softmax(tf.matmul(x,w)+b)#y와 연산을 하는 함수식,activation

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))#loss

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        _,loss_val=sess.run([optimizer,loss],feed_dict={x:x_data,y:y_data})
        # if step % 10==1:
        #     print(loss_val)
        print(loss_val)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    predict = sess.run(hypothesis,feed_dict={x:[[1,2,3,4]]})
    print(predict,sess.run(tf.argmax(predict,axis=1)))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = np.arange(1,13)
    data = data.reshape(3,4)
    predict = sess.run(hypothesis,feed_dict={x:data})
    print(predict,sess.run(tf.argmax(predict,axis=1)))





               
