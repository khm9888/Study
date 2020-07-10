import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts

# 데이터 입력
dataset = load_iris()

x_data=dataset.data
y_data=dataset.target

print(x_data.shape)
print(y_data.shape)

# (150, 4)
# (150,)

''' x_train,y_train,x_test,y_test = tts(x_data,y_data, train_size=0.8)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train,3))
    y_test = sess.run(tf.one_hot(y_test,3))
y_train=y_train.reshape(-1,3)
y_test=y_test.reshape(-1,3) '''

x_train,x_test,y_train,y_test = tts(x_data, y_data, train_size=0.8)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train,3))
    y_test = sess.run(tf.one_hot(y_test,3))
y_train=y_train.reshape(-1,3)
y_test=y_test.reshape(-1,3) 


# y_data = y_data.reshape(y_data.shape[0],1)

x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])

w = tf.Variable(tf.random_normal([4,3]),name='weight') 
b = tf.Variable(tf.random_normal([3]),name='bias')


hypothesis = tf.nn.softmax(tf.matmul(x,w)+b) # matmul - 행렬 연산을 해준다

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
# train = optimizer.minimize(cost)



# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        _,loss_val,hypo_val=sess.run([optimizer,loss,hypothesis],feed_dict={x:x_train,y:y_train})
        # if step % 10==1:
        #     print(loss_val)
        print(loss_val)
        # 실제로 실현되는 부분
    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))
    
    #정확도
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test}))

    # GYU code
    # predicted = tf.arg_max(hypo,1)
    # acc = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     predict = sess.run(hypothesis,feed_dict={x:x_test})
#     print(predict,sess.run(tf.argmax(predict,axis=1)))


# Accuracy: 0.96666664




               
