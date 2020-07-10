import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

# 데이터 입력
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

print(x_train.shape)#(50000, 32, 32, 3)
print(y_train.shape)#(50000, 1)

#전처리1) - minmax (reshape 미필요)
x_train =x_train/255
x_test =x_test/255

#전처리2) to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

learning_rate=0.07
traing_epochs=10
batch_size=100
total_batch = x_train.shape[0]//batch_size #600

x = tf.placeholder(tf.float32, shape=[None,32,32,3])
x_img = tf.reshape(x,[-1,32,32,3])
y = tf.placeholder(tf.float32, shape=[None,10])

w = tf.get_variable("weight1",[3,3,3,32])# (32,(3,3),input=(28,28,3))
layer = tf.nn.conv2d(x_img,w,strides=[1,1,1,1],padding="SAME")
layer = tf.nn.selu(layer)
layer = tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

w = tf.get_variable("weight2",[3,3,32,64])# (32,(3,3),input=(28,28,1))
layer = tf.nn.conv2d(layer,w,strides=[1,1,1,1],padding="SAME")
layer = tf.nn.selu(layer)
layer = tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

# print(layer)

# Tensor("MaxPool_1:0", shape=(?, 8, 8, 64), dtype=float32)
layer = tf.reshape(layer,[-1,8*8*64])

w = tf.get_variable("weight3",[8*8*64,10],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([10]))
layer = tf.nn.softmax(tf.matmul(layer,w)+b)

hypothesis = layer

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
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
            feed_dict = {x:batch_xs,y:batch_ys}
            _,loss_val=sess.run([optimizer,loss],feed_dict=feed_dict)
            avg_cost+=loss_val/total_batch

        print(f"epoch:{epoch+1},loss_val:{avg_cost}")
            # for i in range(total_batch):#600
    name=__file__.split('\\')[-1]
    print(f"{name}")
    print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test}))


'''
keras60_cifar10_CNN
loss:1.049088060259819
acc:0.6744999885559082
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[5 1 0 0 6 6 1 6 3 1 0 9 5 7 9 8 5 5 8 6]
'''

'''
epoch:1,loss_val:1.6925695781707748
epoch:2,loss_val:1.3410056027173998
epoch:3,loss_val:1.1825909184217458
epoch:4,loss_val:1.1006423637866967
epoch:5,loss_val:1.0431726849079135
epoch:6,loss_val:0.9961396683454511
epoch:7,loss_val:0.9556314920186999
epoch:8,loss_val:0.9194904828071596
epoch:9,loss_val:0.8862919116020197
epoch:10,loss_val:0.8563543459177012
d:\Study\tf\tf18_2_cifar_cnn.py
Accuracy: 0.6525

'''