import tensorflow as tf

def s_run(value):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return value.eval()

# tf.set_random_seed(777)
w= tf.Variable(tf.random_normal([1]),name="weight")
b= tf.Variable(tf.random_normal([1]),name="bias")

print(s_run(w))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
aaa = w.eval()
print(aaa)
sess.close()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(w)
print(aaa)
sess.close()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = w.eval(session=sess)
print(aaa)
sess.close()