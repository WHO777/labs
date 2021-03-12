import numpy as np
import argparse
import tensorflow as tf
import glob
import time
import tensorflow_datasets as tfds
from tensorflow.python import keras as keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing

LOG_DIR = 'test_logs'
BATCH_SIZE = 256
NUM_CLASSES = 20
RESIZE_TO = 224


def parse_proto_example(proto):
  keys_to_features = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/label': tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }
  example = tf.io.parse_single_example(proto, keys_to_features)
  example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
  example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
  example['image'] = tf.image.resize(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
  return example['image'], tf.one_hot(example['image/label'], depth=NUM_CLASSES)


def normalize(image, label):
  return tf.image.per_image_standardization(image), label

def build_model():
  inputs = tf.keras.layers.Input(shape=(224, 224, 3))
  model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
  model.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(model.output)     
  x = tf.keras.layers.Flatten()(x)
  #x = tf.keras.layers.BatchNormalization()(x)
  #x = tf.keras.layers.Dropout(0.2)(x)
  #x = tf.keras.layers.Dense(100, activation=tf.keras.layers.ReLU())(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def main():
  args = argparse.ArgumentParser()
  args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files, use single quote to escape *')
  args = args.parse_args()

  dataset_name = "imagewang"
  (ds_train, ds_validation), ds_info = tfds.load(dataset_name, split=["train", "validation"], with_info=True, as_supervised=True, data_dir=args.train) 
  size = (RESIZE_TO, RESIZE_TO)

  ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
  ds_validation = ds_validation.map(lambda image, label: (tf.image.resize(image, size), label))

  ds_train = ds_train.map(input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_validation = ds_validation.map(input_preprocess)
  ds_validation = ds_validation.batch(batch_size=BATCH_SIZE, drop_remainder=True)

  model = build_model()
  print(model.summary())

  model.compile(
    optimizer=tf.optimizers.Adam(lr=1e-3),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy],
  )
  model.fit(
        ds_train,
        epochs=50,
        validation_data=ds_validation,
        callbacks=[
          tf.keras.callbacks.TensorBoard(log_dir),
        ], verbose=2
      )

if __name__ == '__main__':
      main()
