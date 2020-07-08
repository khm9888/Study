import tensorflow as tf

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)


# def s_run(i):
#     with tf.Session() as sess:
#         return sess.run(i)

sess=tf.Session()
s_run = sess.run

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a+b

print(s_run(a+b,feed_dict={a:3,b:4.5}))
print(s_run(a+b,feed_dict={a:[1,3],b:[2,4]}))

add_and_triple = (a+b)*3

print(s_run((a+b)*3,feed_dict={a:3,b:4}))