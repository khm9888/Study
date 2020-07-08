# 넌 정말 sigmoid
# 난 정말 sigmoid

import tensorflow as tf
tf.set_random_seed(777) 

x_data = [[1, 2],
          [2, 3], 
          [3, 1],
          [4, 3],  
          [5, 3],
          [6, 2]]

y_data = [[0], 
          [0], 
          [0],
          [1],
          [1],
          [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])          
y = tf.placeholder(tf.float32, shape=[None, 1])          

# w는 x와 곱해지는 연산의 대상, 따라서 행렬의 곱셈 법칙에 따라  x 데이터의 열과 같은 수의 행으로 설정해주어야 한다
w = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# tf.matmul : 행렬의 곱셈을 해 주는 함수
# x = 5 by 3, w = 3 by 1 
# x * w = 5 by 1
# wx + b = 5 by 1 + 1 = 5 by 1
hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # wx+b

# activation : 결괏값을 활성화함수에 통과시켜서 다음 레이어에 넘겨줌
# tensorflow에서는 활성화함수를 hypothesis에 wrap 해 주면 된다

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))
# sigmoid : 분류 쪽에서 사용하는 활성화 함수
# sigmoid 사용하는 cost 정의?
# sigmoid 를 사용했을 때의 cost의 식은 이러하다, 정도까지만 이해
# binarycross_entropy 식이다?
# activation 함수에 대한 구현

# 현재 hidden layer 는 없는 케라스 모델과 같다고 생각하면 된다

# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00000045)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.09)

# lr = 0.1 , epo = 5000
# 5000 cost : 0.03828542
# Accuracy
# 1.0

# lr = 0.9, epo = 5000
# 5000 cost : 0.0046946993
# acc = 1.0

# lr = 0.9, epo = 1000
# 1000 cost : 0.022052964
# acc = 1.0

# lr = 0.9, epo = 100
# 100 cost : 0.15251936
# acc = 1.0

# lr = 0.9, epo = 50
# 50 cost : 0.30314392
# acc = 1.0

# lr = 0.1, epo = 1000
# 1000 cost : 0.01988281
# acc = 1.0

# lr = 0.1, epo = 500
# 500 cost : 0.23859692
# acc = 1.0

# lr = 0.09, epo = 500
# 500 cost : 0.25337335
# acc = 1.0


train = optimizer.minimize(cost)

# 0.5 초과하면 1, 0.5 이하면 0 -> sigmoid
# tf.cast() : true/ false를 1.0 / 0.0 으로 반환
# hypothesis > 0.5 == true == 1.0 반환
# hypothesis =< 0.5 == false == 0.0 반환
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))
# predicted(0 아니면 1) 과 y가 같냐, tf.equal로 비교
#  tf.equal(predicted,Y) 가 같으면 1 다르면 0, 이를 더해서 평균을 낸다.
# 같으면 같을 수록 1이 많고 -> 평균이 1에 가까워짐 -> ACC가 높음


# with 아래 주석처리하고 실행하면 prerdicted, accuracy가 계산이 될까?
# Nope
# Session을 통과하지 않았기 때문에
# 실질적으로  with 문 안의 sess.run 이 실행되어야 계산 된다
# predicted, accuracy 를 정의 해 놓은 것 까지는 그냥 'Ready' 상태라고 보면 된다
# 모든 것은 sess.run을 통과해야 시작되는 것

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # 변수 초기화 : 0으로 만드는 것이 아니라, 변수를 선언하는 것이라고 받아들여라

    for step in range(501) :
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})

        if step % 10 == 0 :
            print(step, "cost :", cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_data, y:y_data})
    # print("\n Hypothesis :", h, "\n Correct (y) :", c, "\n Accuracy :", a)
    print("-------------------------")
    print("Hypothesis")
    print(h)
    print("-------------------------")
    print("Correct(y)")
    print(c)
    print("-------------------------")
    print("Accuracy")
    print(a)


# 구체적으로 세세하게 모든 걸 알기는 어려움
# 전체적으로 케라스로 만들 때와 비교하면서 받아들이고 익숙해져야 함