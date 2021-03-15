import numpy as np
import argparse
import tensorflow as tf
import glob
import time
import tensorflow_datasets as tfds
from tensorflow.python import keras as keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing

LOG_DIR = 'logs'
BATCH_SIZE = 64
NUM_CLASSES = 20
RESIZE_TO = 224


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
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
  
  '''args = argparse.ArgumentParser()
  args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files, use single quote to escape *')
  args = args.parse_args()

  ds_train = tf.keras.preprocessing.image_dataset_from_directory(
      args.train, labels='inferred',
       color_mode='rgb', batch_size=BATCH_SIZE, image_size=(RESIZE_TO,
      RESIZE_TO), shuffle=True, seed=13, validation_split=0.3, subset="training"
  )

  ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
      args.train, labels='inferred',
       color_mode='rgb', batch_size=BATCH_SIZE, image_size=(RESIZE_TO,
      RESIZE_TO), shuffle=True, seed=13, validation_split=0.3, subset="validation"
  )
  #ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
  #ds_validation = ds_validation.map(lambda image, label: (tf.image.resize(image, size), label))

  ds_train = ds_train.map(input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_validation = ds_validation.map(input_preprocess)

  model = build_model()
  print(model.summary())

  model.compile(
    optimizer=tf.optimizers.Adam(lr=1e-3),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy],
  )'''
  '''model.fit(
        ds_train,
        epochs=50,
        validation_data=ds_validation,
        callbacks=[
          tf.keras.callbacks.TensorBoard(LOG_DIR),
        ]
      )'''

if __name__ == '__main__':
      main()
