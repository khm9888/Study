import tensorflow as tf
import numpy as np

# data = hihello
idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1, 1)
print(_data.shape)      #(7, 1)
print(_data)
print(type(_data))

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()

print("=================================================")
print(_data)
print(type(_data))
print(_data.dtype)

x_data = _data[:6,]     # hihell
y_data = _data[1:,]     # ihello

print("============x============")
print(x_data)
print("============y============")
print(y_data)

print("=========== y argmax ============")
y_data = np.argmax(y_data, axis=1)
print(y_data)
print(y_data.shape)

x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)

print(x_data.shape)
print(y_data.shape)

sequence_length = 6
input_dim = 5
output = 5
batch_size = 1 # 전체 행

# X = tf.placeholder(tf.float32, (None, sequence_length, input_dim))
# Y = tf.placeholder(tf.float32, (None, sequence_length))

X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
# Y = tf.compat.v1.placeholder(tf.float32, (None, sequence_length))
Y = tf.compat.v1.placeholder(tf.int32, (None, sequence_length))

print(X) # 텐서의 형태 출력 / 값 나올라면 sess.run
print(Y)

# 2. 모델 구성
# model.add(LSTM(output, input_dim=(6, 5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
                            # model.add(LSTM)
print(hypothesis)   # (?, 6, 5)

# 3-1. 컴파일
weights = tf.ones([batch_size, sequence_length])    # [1, 6]
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets = Y, weights = weights)
cost = tf.reduce_mean(sequence_loss)

# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
print("끗")

prediction = tf.argmax(hypothesis, axis=2)
print(prediction)


# 3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        print(i, "loss : ", loss, "prediction : ", result, "true Y :", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPrediction str : ", ''.join(result_str))

