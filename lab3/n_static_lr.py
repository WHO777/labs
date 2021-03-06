"""This module implements data feeding and training loop to create model
to classify X-Ray chest images as a lab example for BSU students.
"""

__author__ = 'Alexander Soroka, soroka.a.m@gmail.com'
__copyright__ = """Copyright 2020 Alexander Soroka"""


import argparse
import glob
import numpy as np
import tensorflow as tf
import time
import math
from math import exp
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.python import keras as keras
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import image


# Avoid greedy memory allocation to allow shared GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


LOG_DIR = 'logs'
BATCH_SIZE = 64
NUM_CLASSES = 20
RESIZE_TO = 224
TRAIN_SIZE = 12786

def parse_proto_example(proto):
  keys_to_features = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/label': tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }
  example = tf.io.parse_single_example(proto, keys_to_features)
  example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
  example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.uint8)
  example['image'] = tf.image.resize(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
  return example['image'], tf.one_hot(example['image/label'], depth=NUM_CLASSES)


def normalize(image, label):
  return tf.image.per_image_standardization(image), label


def create_dataset(filenames, batch_size):
  """Create dataset from tfrecords file
  :tfrecords_files: Mask to collect tfrecords file of dataset
  :returns: tf.data.Dataset
  """
  return tf.data.TFRecordDataset(filenames)\
    .map(parse_proto_example, num_parallel_calls=tf.data.AUTOTUNE)\
    .cache()\
    .batch(batch_size)\
    .prefetch(tf.data.AUTOTUNE)


def build_model():
  inputs = tf.keras.layers.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
  model.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(model.output)     
  #x = tf.keras.layers.BatchNormalization()(x)
  #x = tf.keras.layers.Dropout(0.2)(x)
  #x = tf.keras.layers.Dense(100, activation=tf.keras.layers.ReLU())(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def main():
  args = argparse.ArgumentParser()
  args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files, use single quote to escape *')
  args = args.parse_args()

  dataset = create_dataset(glob.glob(args.train), BATCH_SIZE)
  train_size = int(TRAIN_SIZE * 0.7 / BATCH_SIZE)
  train_dataset = dataset.take(train_size)
  validation_dataset = dataset.skip(train_size)
 

  k = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  drop = [0.99, 0.95, 0.5, 0.4, 0.35, 0.3, 0.1]
  epochs_drop = [1, 2, 10, 7, 7, 5, 15]

  for i in range(7):

    exp_sheduler = lambda epoch: 0.1 * math.exp(-k[i]*epoch)
    step_sheduler = lambda epoch: 0.1 * math.pow(drop[i],  
           math.floor((1+epoch)/epochs_drop[i]))

    model = build_model()
    model.compile(
      optimizer=tf.optimizers.Adam(),
      loss=tf.keras.losses.categorical_crossentropy,
      metrics=[tf.keras.metrics.categorical_accuracy],
    )

    log_dir='{}/exp_k{}'.format(LOG_DIR, k[i])
    model.fit(
      train_dataset,
      epochs=50,
      validation_data=validation_dataset,
      callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir),
        tf.keras.callbacks.LearningRateScheduler(exp_sheduler)
      ]
    )

    log_dir='{}/step_drop{}_epochs{}'.format(LOG_DIR, drop[i], epochs_drop[i])
    model.fit(
      train_dataset,
      epochs=50,
      validation_data=validation_dataset,
      callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir),
        tf.keras.callbacks.LearningRateScheduler(step_sheduler)
      ]
    )


if __name__ == '__main__':
    main()
