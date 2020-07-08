import tensorflow as tf
tf.set_random_seed(777)


x_train = [1,2,3]
y_train = [1,2,3]

w= tf.Variable(tf.random_normal([1]),name="weight")
b= tf.Variable(tf.random_normal([1]),name="bias")

#sess 줄이기

sess=tf.Session()
s_run = sess.run

# s_run(tf.global_variables_initializer())
# print(s_run(w))

hypothesis=x_train*w+b

cost = tf.reduce_mean(tf.square(hypothesis-y_train))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        _, cost_val,w_val,b_val = sess.run([train,cost,w,b])
        if step%20==0:
            print(step,cost_val,w_val,b_val)


