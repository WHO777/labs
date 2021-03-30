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
import os
import math
import albumentations as A
import matplotlib.pyplot as plt
from functools import partial
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
  example['image'] = tf.image.resize(example['image'], tf.constant(286, 286]), method='nearest')
  return example['image'], tf.one_hot(example['image/label'], depth=NUM_CLASSES)

  
def aug_fn(image, label, transforms):
  
    def BrightnessContrast(image):
      data = {"image":image}
      aug_data = transforms(**data)
      aug_img = aug_data["image"]
      #aug_img = tf.image.resize(aug_img, size=[RESIZE_TO, RESIZE_TO])
      aug_img = tf.cast(aug_img, tf.uint8)
      return aug_img

    aug_image = tf.numpy_function(func=BrightnessContrast, inp=[image], Tout=(tf.uint8))
    return aug_image, label


def set_shapes(img, label, img_shape=(RESIZE_TO, RESIZE_TO, 3)):
    img.set_shape(img_shape)
    return img, label



def create_dataset(filenames, batch_size, transforms):
  """Create dataset from tfrecords file
  :tfrecords_files: Mask to collect tfrecords file of dataset
  :returns: tf.data.Dataset
  """
  return tf.data.TFRecordDataset(filenames)\
    .map(parse_proto_example, num_parallel_calls=tf.data.AUTOTUNE)\
    .map(partial(aug_fn, transforms=transforms), num_parallel_calls=tf.data.AUTOTUNE)\
    .map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(BATCH_SIZE)\
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
  
  sheduler = lambda epoch: 0.01 * math.exp(-0.3*epoch)

  width, height, p = 224, 224, 1
  transforms = A.Compose([
      A.RandomCrop(width, height, p=p),
   ])
  dataset = create_dataset(glob.glob(args.train), BATCH_SIZE, transforms)
  
  for i, (x, y) in enumerate(dataset.take(10)):
    print(x[i].shape)
    plt.imshow(x[i])
    output_path = os.path.join('examples/RandomCrop/',str(i)+'.jpg')            
    plt.savefig(output_path)

  train_size = int(TRAIN_SIZE * 0.7 / BATCH_SIZE)
  train_dataset = dataset.take(train_size)
  validation_dataset = dataset.skip(train_size)

  model = build_model()
  model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy],
   )

  log_dir='{}/RandomCrop_h{}_w{}_p{}_s286'.format(LOG_DIR, height, width, p)
  print(log_dir)
  model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir),
      tf.keras.callbacks.LearningRateScheduler(sheduler),
    ]
  )


if __name__ == '__main__':
    main()
