#3+4+5
#4-3
#3*4
#4/2
import tensorflow as tf

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(5.0)
node4 = tf.constant(2.0)

add_value = tf.add_n([node1,node2,node3])
subtract_value=tf.subtract(node2,node1)
multiply_value=tf.multiply(node1,node2)
divide_value=tf.divide(node2,node4)

sess=tf.Session()
s_run = sess.run

print(s_run(add_value))
print(s_run(subtract_value))
print(s_run(multiply_value))
print(s_run(divide_value))