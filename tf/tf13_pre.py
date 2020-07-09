from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
import tensorflow as tf
import numpy as np

def min_max_scaler(dataset):
    numerator = dataset-np.min(dataset,0)
    denominator = np.max(dataset,0)-np.min(dataset,0)
    return numerator/(denominator+1e-7)

dataset=xy = np.array(

    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

dataset = min_max_scaler(dataset)
print()

x_data = dataset[:,:-1]
y_data = dataset[:,[-1]]

x_train,y_train,x_test,y_test = tts(x_data,y_data,train_size=0.7)

print(x_data.shape)
print(y_data.shape)

y_data = y_data.reshape(y_data.shape[0],1)

x=tf.placeholder(tf.float32,shape=[None,4])
y=tf.placeholder(tf.float32,shape=[None,1])
w = tf.Variable(tf.random_normal([4,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothesis = tf.matmul(x,w)+b

loss = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)

train = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        loss_val,hy_val,_ = sess.run([loss,hypothesis,train],feed_dict={x:x_data,y:y_data})
        # print(type(hy_val))
        # print(hy_val.shape[0])
        if step%100==1:
            print(f"step:{step},cost_val:{loss_val}")
            for i in range(8):
                print(f"hy_val_{i}:{hy_val[i]}  y_data_{i}:{y_data[i]}")
    
    hypo_val=sess.run([hypothesis],feed_dict={x:x_test})

    r2 = sess.run(r2_score(hypo_val,y_test))
    print(r2)