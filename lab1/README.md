#h1 Архитектура
```python 
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
```
## h2 Conv2D
```Conv2D``` представляет собой свертку 3х3 с ```padding = 0``` и ```strides = (1,1)```
## h2 MaxPool2D
```MaxPool2D``` выбирает из 4(padding=0, stride=0, size=(2,2)) 'пикселей' один с максимальным значением, тем самым уменьшая исходную размерность в 4 раза
## h2 Flatten
```Flatten``` делает reshape из размерности (8, 111, 111) в (None, 98568)
# h1 Графика
Синяя - validation
Оранжевая - train
## h2 epoch categorical accuracy
![epoch_categorical_accuracy_1_](https://user-images.githubusercontent.com/61012068/110214611-719a0900-7eb6-11eb-94e9-92f996a417a2.jpg)
## h2 epoch loss
![epoch_loss_1_](https://user-images.githubusercontent.com/61012068/110214616-78c11700-7eb6-11eb-81d0-7595447c5c91.jpg)


