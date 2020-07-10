import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

# 데이터 입력
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)#(60000, 28, 28)
print(y_train.shape)#(60000,)

#전처리1) - minmax
x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])/255
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])/255

#전처리2) to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

learning_rate=0.001
traing_epochs=35
batch_size=100
total_batch = x_train.shape[0]//batch_size #600

print()

x = tf.placeholder(tf.float32, shape=[None,28*28])
y = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32)

# w = tf.Variable(tf.zeros([28*28,10]),name="weight")
w = tf.get_variable("weight1",[28*28,512],initializer=tf.contrib.layers.xavier_initializer())
#<tf.Variable 'weight1:0' shape=(784, 512) dtype=float32_ref>
b = tf.Variable(tf.zeros([512]),name="bias1")
# <tf.Variable 'bias1:0' shape=(512,) dtype=float32_ref>
layer = tf.nn.selu(tf.matmul(x,w)+b)
# Tensor("Selu:0", shape=(?, 512), dtype=float32)
layer = tf.nn.dropout(layer,keep_prob=keep_prob)
# Tensor("dropout/mul_1:0", shape=(?, 512), dtype=float32)


# tf.contrib.layers.
w = tf.get_variable("weight2",[512,512],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([512]),name="bias2")
layer = tf.nn.selu(tf.matmul(layer,w)+b)
layer = tf.nn.dropout(layer,keep_prob=keep_prob)

w = tf.get_variable("weight3",[512,128],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([128]),name="bias3")
layer = tf.nn.selu(tf.matmul(layer,w)+b)
layer = tf.nn.dropout(layer,keep_prob=keep_prob)

w = tf.get_variable("weight4",[128,10],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([10]),name="bias4")
layer = tf.nn.softmax(tf.matmul(layer,w)+b)

hypothesis = layer

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))
#정확도
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(traing_epochs):#15
        avg_cost=0

        for i in range(total_batch):
            order=i*batch_size
            batch_xs, batch_ys = x_train[order:+order+batch_size], y_train[order:order+batch_size]
            feed_dict = {x:batch_xs,y:batch_ys,keep_prob:0.7}
            _,loss_val=sess.run([optimizer,loss],feed_dict=feed_dict)
            avg_cost+=loss_val/total_batch

        print(f"epoch:{epoch+1},loss_val:{avg_cost}")
            # for i in range(total_batch):#600


    print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test,keep_prob:1}))


# epoch:33,loss_val:0.08123232854297385
# epoch:34,loss_val:0.08043446005632474
# epoch:35,loss_val:0.0821741102013039
# Accuracy: 0.964