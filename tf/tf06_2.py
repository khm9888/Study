import tensorflow as tf

def s_run(value):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return value.eval()
        
#데이터 입력
# x_train = [1,2,3]
x_train = tf.constant([1,2,3])
# y_train = [1,2,3]
y_train = tf.constant([1,2,3])

x_train_h = tf.placeholder(tf.float32)
y_train_h = tf.placeholder(tf.float32)
c = tf.placeholder(tf.float32)
d = tf.placeholder(tf.float32)

tf.set_random_seed(777)
w= tf.Variable(tf.random_normal([1]),name="weight")
b= tf.Variable(tf.random_normal([1]),name="bias")

hypothesis=x_train_h*w+b

#모델 훈련
cost = tf.reduce_mean(tf.square(hypothesis-y_train_h))

# cost = tf.reduce_mean(tf.square(a-b,feed_dict = {a:hypothesis,b:y_train})) #경사하강법,  loss

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)#optimizer, sgd

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):#epoch
        _, cost_val,w_val,b_val = sess.run([train,cost,w,b],feed_dict={x_train_h:[1,2,3],y_train_h:[1,2,3]})
        if step%20==0:
            print(step,cost_val,w_val,b_val)


