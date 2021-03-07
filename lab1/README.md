

![epoch_categorical_accuracy_1_](https://user-images.githubusercontent.com/61012068/110214611-719a0900-7eb6-11eb-94e9-92f996a417a2.jpg)
![epoch_loss_1_](https://user-images.githubusercontent.com/61012068/110214616-78c11700-7eb6-11eb-81d0-7595447c5c91.jpg)

```python 
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
```
