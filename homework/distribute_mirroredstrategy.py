# Note: tf.distribute.Strategy currently does 
# not support TensorFlow's partitioned variables (where a single variable is split across multiple devices) at this time.
# 현재 TensorFlow의 분할 변수(단일 변수가 여러 장치에 걸쳐 분할되는 경우)는 지원하지 않는다.


# https://www.tensorflow.org/tutorials/distribute/custom_training?hl=ko


# Import TensorFlow
import tensorflow as tf

# 헬퍼 라이브러리들
import numpy as np
import os

print(tf.__version__)



fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# 하나의 차원을 배열에 추가 -> 새로운 shape == (28, 28, 1)
# 이렇게 하는 이유는 우리의 모델에서 첫 번째 층이 합성곱 층이고
# 합성곱 층은 4D 입력을 요구하기 때문입니다.
# (batch_size, height, width, channels).
# batch_size 차원은 나중에 추가할 것입니다.

train_images = train_images[..., None]
test_images = test_images[..., None]

# 이미지를 [0, 1] 범위로 변경하기.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)


# 만약 장치들의 목록이 `tf.distribute.MirroredStrategy` 생성자 안에 명시되어 있지 않다면,
# 자동으로 장치를 인식할 것입니다.
strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.Strategy()


print ('장치의 수: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10


with strategy.scope():

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE) 
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


# 체크포인트들을 저장하기 위해서 체크포인트 디렉토리를 생성합니다.
checkpoint_dir = 'D:\Study\homework\data/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


with strategy.scope():
  # reduction을 `none`으로 설정합니다. 그래서 우리는 축소를 나중에 하고,
  # GLOBAL_BATCH_SIZE로 나눌 수 있습니다.
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)
  # 또는 loss_fn = tf.keras.losses.sparse_categorical_crossentropy를 사용해도 됩니다.
  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')


# 모델과 옵티마이저는 `strategy.scope`에서 만들어져야 합니다.
with strategy.scope():
  model = create_model()

  optimizer = tf.keras.optimizers.Adam()

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

with strategy.scope():
  def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss 

  def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)

with strategy.scope():
  # `experimental_run_v2`는 주어진 계산을 복사하고,
  # 분산된 입력으로 계산을 수행합니다.
  
  @tf.function
  def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.experimental_run_v2(train_step,
                                                      args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)
 
  @tf.function
  def distributed_test_step(dataset_inputs):
    return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

  for epoch in range(EPOCHS):
    # 훈련 루프
    total_loss = 0.0
    num_batches = 0
    for x in train_dist_dataset:
      total_loss += distributed_train_step(x)
      num_batches += 1
    train_loss = total_loss / num_batches

    # 테스트 루프
    for x in test_dist_dataset:
      distributed_test_step(x)

    if epoch % 2 == 0:
      checkpoint.save(checkpoint_prefix)

    template = ("에포크 {}, 손실: {}, 정확도: {}, 테스트 손실: {}, "
                "테스트 정확도: {}")
    print (template.format(epoch+1, train_loss,
                           train_accuracy.result()*100, test_loss.result(),
                           test_accuracy.result()*100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='eval_accuracy')

new_model = create_model()
new_optimizer = tf.keras.optimizers.Adam()

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

@tf.function
def eval_step(images, labels):
  predictions = new_model(images, training=False)
  eval_accuracy(labels, predictions)

checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, model=new_model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for images, labels in test_dataset:
  eval_step(images, labels)

print ('전략을 사용하지 않고, 저장된 모델을 복원한 후의 정확도: {}'.format(
    eval_accuracy.result()*100))

with strategy.scope():
  for _ in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    train_iter = iter(train_dist_dataset)

    for _ in range(10):
      total_loss += distributed_train_step(next(train_iter))
      num_batches += 1
    average_train_loss = total_loss / num_batches

    template = ("에포크 {}, 손실: {}, 정확도: {}")
    print (template.format(epoch+1, average_train_loss, train_accuracy.result()*100))
    train_accuracy.reset_states()

with strategy.scope():
  @tf.function
  def distributed_train_epoch(dataset):
    total_loss = 0.0
    num_batches = 0
    for x in dataset:
      per_replica_losses = strategy.experimental_run_v2(train_step,
                                                        args=(x,))
      total_loss += strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
      num_batches += 1
    return total_loss / tf.cast(num_batches, dtype=tf.float32)

  for epoch in range(EPOCHS):
    train_loss = distributed_train_epoch(train_dist_dataset)

    template = ("Epoch {}, Loss: {}, Accuracy: {}")
    print (template.format(epoch+1, train_loss, train_accuracy.result()*100))

    train_accuracy.reset_states()

    