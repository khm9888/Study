import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts

dataset = np.arange(1,11)
dataset = dataset.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
dataset = encoder.fit_transform(dataset)
print(dataset)
print(dataset.toarray())


# print("_data")
# print(_data)
x_data=list()
y_data=list()

length = 5
for i in range(5):

    xs = dataset[i:i+length]
    ys = dataset[i+length]
    x_data.append(xs)
    y_data.append(ys)

x_data=np.array(x_data)
y_data=np.array(y_data)

print(x_data.shape)#(5, 5, 1)
print(y_data.shape)#(5, 1)

x_data = x_data.reshape(1,5,5)
y_data = y_data.reshape(1,5)

# x_data=x_data.reshape(x_data.shape[0],x_data.shape[1])
# y_data=y_data.reshape(y_data.shape[0],1)

print("="*10)
print(x_data)
print("="*10)
print(y_data)
print("="*10)

print(x_data.shape)#(5, 5, 1)
print(y_data.shape)#(5, 1)

print("y_data")
print(y_data)
sequence_length=5
input_shape=(5,5)
output=100
batch_size=1
traing_epochs=400
total_batch = x_data.shape[0]//batch_size
# x = tf.placeholder(tf.float32,(None,input_shape))
# y = tf.placeholder(tf.float32,(None,sequence_length))

x = tf.compat.v1.placeholder(tf.float32,(None,input_shape[0],sequence_length))
y = tf.compat.v1.placeholder(tf.int32,(None,sequence_length))

#2.모델구성

#model.add(LSTM(output,input_shape=(6,5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _state = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)

weights = tf.compat.v1.ones([batch_size,sequence_length])#1,1

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets=y, weights=weights)
loss = tf.compat.v1.reduce_mean(sequence_loss)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.015).minimize(loss)

correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))
#정확도
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#3.훈련
with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(traing_epochs):#15
        avg_cost=0

        for i in range(total_batch):
            order=i*batch_size
            batch_xs, batch_ys = x_data[order:+order+batch_size], y_data[order:order+batch_size]
            feed_dict = {x:x_data,y:y_data}
            _,loss_val=sess.run([optimizer,loss],feed_dict=feed_dict)
            avg_cost+=loss_val/total_batch


        if epoch%100==0:
            print(f"{epoch} avg_cost :{avg_cost}")
    print("Accuracy:",sess.run(accuracy,feed_dict={x:x_data,y:y_data}))

