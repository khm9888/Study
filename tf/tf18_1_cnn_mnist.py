import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

# 데이터 입력
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)#(60000, 28, 28)
print(y_train.shape)#(60000,)

def min_max_scaler(dataset):
    numerator = dataset-np.min(dataset,0)
    denominator = np.max(dataset,0)-np.min(dataset,0)
    return numerator/(denominator+1e-7)

#전처리1) - minmax
x_train = min_max_scaler(x_train.reshape(-1,x_train.shape[1],x_train.shape[2],1))
x_test = min_max_scaler(x_test.reshape(-1,x_test.shape[1],x_test.shape[2],1))

#전처리2) to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

learning_rate=0.1
traing_epochs=24
batch_size=100
total_batch = x_train.shape[0]//batch_size #600

x = tf.placeholder(tf.float32, shape=[None,28,28,1])
x_img = tf.reshape(x,[-1,28,28,1])
y = tf.placeholder(tf.float32, shape=[None,10])
# keep_prob = tf.placeholder(tf.float32)

w = tf.get_variable("weight1",[3,3,1,32])# (32,(3,3),input=(28,28,1))
layer = tf.nn.conv2d(x_img,w,strides=[1,1,1,1],padding="SAME")
layer = tf.nn.selu(layer)
layer = tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

w = tf.get_variable("weight2",[3,3,32,64])# (32,(3,3),input=(28,28,1))
layer = tf.nn.conv2d(layer,w,strides=[1,1,1,1],padding="SAME")
layer = tf.nn.selu(layer)
layer = tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

layer = tf.reshape(layer,[-1,7*7*64])

w = tf.get_variable("weight3",[7*7*64,10],initializer=tf.contrib.layers.xavier_initializer())
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

    print(f"{__file__.split('\\')[-1]}")
    print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test}))


# epoch:23,loss_val:0.006594989188688486
# epoch:24,loss_val:0.005967014006970807
# Accuracy: 0.9869