import tensorflow as tf

# print(tf.__version__)

# hello = tf.constant("Hello world")
# print(hello)

# sess = tf.Session()
# print(sess.run(hello))


a = tf.constant(2); b = tf.constant(3); c = tf.multiply(a, b)


def s_run(value):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # return value.eval()
        return tf.constant(value).eval()



print(s_run(1)) 
