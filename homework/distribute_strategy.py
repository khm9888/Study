from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version 기능은 코랩에서만 사용할 수 있습니다.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

BUFFER_SIZE = 10000
BATCH_SIZE = 64

# MNIST 데이터를 (0, 255] 범위에서 (0., 1.] 범위로 조정
def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

datasets, info = tfds.load(name='mnist',
                           with_info=True,
                           as_supervised=True)

train_datasets_unbatched = datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)
train_datasets = train_datasets_unbatched.batch(BATCH_SIZE)

def build_and_compile_cnn_model():
      model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model


single_worker_model = build_and_compile_cnn_model()
single_worker_model.fit(x=train_datasets, epochs=3)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})










