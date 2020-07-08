import tensorflow as tf

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)

node3 = tf.add(node1,node2)

print(f"node1:{node1.Const}")
print(f"node2:{node2}")
print(f"node3:{node3}")