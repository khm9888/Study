import tensorflow as tf
import matplotlib.pyplot as plt


# def s_run(value,**feed_dict):
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         return sess.run(value,feed_dict)

x=list(map(float,[1,2,3]))
y=list(map(float,[1,2,3]))

w= tf.placeholder(tf.float32)
# b= [1]


hypothesis=x*w

#모델 훈련
cost = tf.reduce_mean(tf.square(hypothesis-y))

# cost = tf.reduce_mean(tf.square(a-b,feed_dict = {a:hypothesis,b:y_train})) #경사하강법,  loss
w_history=[]

cost_history=[]

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)#optimizer, sgd
with tf.Session() as sess:
    for i in range(-30,50):
        curr_w =i*0.1
        curr_cost =sess.run(cost,feed_dict={w:curr_w})
        w_history.append(curr_w)
        cost_history.append(curr_cost)

plt.plot(w_history,cost_history)
plt.show()