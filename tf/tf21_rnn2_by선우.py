import numpy as np
import tensorflow as tf

dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(dataset.shape)        # (10,)

# # RNN 모델을 짜시오!
def split_x(seq, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

# 데이터 자르기
data = split_x(dataset, 5)
print(data.shape)           # (6, 5)
print(data)

x_data = data[:, :4]
y_data = data[:, 4:]
print(x_data.shape)         # (6, 4)
print(y_data.shape)         # (6, 1)

x_data = x_data.reshape(1, 6, 4)
y_data = y_data.reshape(1, 6)
print(x_data.shape)         # (1, 6, 4)
print(y_data.shape)         # (1, 6)

## 모델 구성
sequence_length = 6
input_dim = 4
output = 100
batch_size = 1
EPOCH = 400

X = tf.compat.v1.placeholder(tf.float32, shape = (None, sequence_length, input_dim))
Y = tf.compat.v1.placeholder(tf.int64, shape = (None, sequence_length))
print(X)                    # Tensor("Placeholder:0", shape=(?, 6, 4), dtype=float32)
print(Y)                    # Tensor("Placeholder_1:0", shape=(?, 6), dtype=int64)

_lstm = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(_lstm, X, dtype = tf.float32)
print(hypothesis)           # Tensor("rnn/transpose_1:0", shape=(?, 6, 1), dtype=float32)

## 컴파일
weight = tf.compat.v1.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets = Y, weights = weight)
cost = tf.compat.v1.reduce_mean(sequence_loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

prediction = tf.math.argmax(hypothesis, axis = 2)
print(f'Prediction : {prediction}')

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict = {X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict = {X: x_data})
        print(f'\nEpoch : {i}, Prediction : {result}, true Y : {y_data}')
        print(f'\nLoss : {loss}')