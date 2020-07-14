import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts


#hihello
text = "hihello"

text_list=[i for i in text]
text = list(set(text_list))
text.sort()
# print(text)

idx2char = text

_data = np.array(["h","i","h","e","l","l","o"],dtype=np.str)
print(_data.shape)
_data = _data.reshape(-1,1)
print(_data)
print(type(_data))

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

_data = encoder.fit_transform(_data).toarray()

# print("_data")
# print(_data)

x_data = _data[:-1]
y_data = _data[1:]
print("="*10)
print(x_data)
print("="*10)
print(y_data)
print("="*10)

y_data  = np.argmax(y_data,1)

x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)

print(x_data.shape)
sequence_length=6
input_shape=(6,5)
output=5
batch_size=1
traing_epochs=400
total_batch = x_data.shape[0]//batch_size
# x = tf.placeholder(tf.float32,(None,input_shape))
# y = tf.placeholder(tf.float32,(None,sequence_length))

x = tf.placeholder(tf.float32,(None,input_shape[0],input_shape[1]))
y = tf.placeholder(tf.int32,(None,sequence_length))

#2.모델구성

#model.add(LSTM(output,input_shape=(6,5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _state = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)

weights = tf.ones([batch_size,sequence_length])#1,6

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis,targets=y,weights=weights)
loss = tf.reduce_mean(sequence_loss)

optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(hypothesis,2)



#3.훈련

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(traing_epochs):#15
        avg_cost=0

        for i in range(total_batch):
            order=i*batch_size
            batch_xs, batch_ys = x_data[order:+order+batch_size], y_data[order:order+batch_size]
            feed_dict = {x:x_data,y:y_data}
            _,loss_val=sess.run([optimizer,loss],feed_dict=feed_dict)
            avg_cost+=loss_val/total_batch
        result = sess.run(prediction,feed_dict={x:x_data})

        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(f"{epoch} avg_cost :{avg_cost}")
        print(f"pediction_str : {''.join(result_str)}")
