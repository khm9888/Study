import tensorflow as tf

x1_data=list(map(float,list(range(73,120,10))))
x2_data=list(map(float,list(range(75,120,10))))
x3_data=list(map(float,list(range(15,65,10))))
y_data=list(map(float,list(range(100,150,10))))

# print(x1_data)

x1=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)
x3=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]),name="weight1")
w2 = tf.Variable(tf.random_normal([1]),name="weight2")
w3 = tf.Variable(tf.random_normal([1]),name="weight3")
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothesis = x1*w1+x2*w2+x3*w3+b
cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_val,_=sess.run([cost,hypothesis,train],feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
        if step%10==1:
            print(f"step:{step},cost_val:{cost_val},hy_val:{hy_val}")