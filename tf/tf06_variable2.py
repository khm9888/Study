#hypothesis 를 구하시오.

import tensorflow as tf

def s_run(value):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return value.eval()
        # return sess.run(value)

# x_train = tf.placeholder(tf.float32)
x_train = [1,2,3]
# y_train = tf.placeholder(tf.float32)

# tf.set_random_seed(777)
w= tf.Variable(tf.constant([1,2,3],dtype=tf.float32),name="weight")
b= tf.Variable(tf.random_normal([1]),name="bias")

hypothesis = x_train*w+b

print(s_run(hypothesis))

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# aaa = w.eval()
# print(aaa)
# sess.close()


# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# aaa = sess.run(w)
# print(aaa)
# sess.close()


# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# aaa = w.eval(session=sess)
# print(aaa)
# sess.close()