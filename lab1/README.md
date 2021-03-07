# Архитектура начальной сети
```python 
inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
```
## Conv2D
```Conv2D``` представляет собой свертку 3х3 с ```padding = 0``` и ```strides = (1,1)```
## MaxPool2D
```MaxPool2D``` выбирает из 4(```padding=0, stride=0, size=(2,2)```) 'пикселей' один с максимальным значением, тем самым уменьшая исходную размерность в 4 раза
## Flatten
```Flatten``` делает reshape из размерности ```(8, 111, 111)``` в ```(None, 98568)```
## Dense
```Dense``` представляет собой полносвязный слой с размерностью ```(None, 98568)``` на входе и ```(None, 20)``` на выходе <br/>
За ним следует функция акивации ```softmax``` которая превращает выходы из ```Dense``` в вероятности 
# Графика
Синяя - validation <br/>
Оранжевая - train
## epoch categorical accuracy
![epoch_categorical_accuracy_1_](https://user-images.githubusercontent.com/61012068/110214611-719a0900-7eb6-11eb-94e9-92f996a417a2.jpg)
## epoch loss
![epoch_loss_1_](https://user-images.githubusercontent.com/61012068/110214616-78c11700-7eb6-11eb-81d0-7595447c5c91.jpg)
***
# Архитектура модифицированной сети 
```python
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation=tf.keras.layers.ReLU())(inputs)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.keras.layers.ReLU())(x)
  x = tf.keras.layers.MaxPool2D()(x) 
  x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.keras.layers.ReLU())(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.keras.layers.ReLU())(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(94, activation=tf.keras.layers.ReLU())(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
```
## ReLU
```ReLU``` представляет собой функцию активации которая нужна для придания нелинейности нейронной сети
В нейронную сеть были добавлены 3 ```Conv2D``` слоя чтобы увеличить возможность выразить более сложные признаки, этим обуславливается и наличие большего числа фильтров в данных слоях,а именно ```16, 32, 64```, благодаря этим изменениям сеть может различать признаки более сложные чем линии,градиенты, и т.п. а именно их комбинации. Так же увеличение количества фильтров с каждым слоем показывает себя лучше чем статическое их значение, так как разничных комбинайций более простых признаков становится все больше.В архитектуру так же был добавлен дополнительный полносвязный слой для лучшего разделения на классы. В качестве нелинейности выбрана функция ```ReLU```.
# Анализ 
