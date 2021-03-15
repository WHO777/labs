Решение задачи классификации изображений из набора данных Oregon Wildlife с
использованием нейронных сетей глубокого обучения и техники обучения Transfer
Learning

# 1. Cлучайное начальное приближение
## Архитектура 
```python
inputs = tf.keras.layers.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
return tf.keras.Model(inputs=inputs, outputs=outputs)
```
***
## Визуализация обучения
Синяя - данные для валидации(проверки качества) <br/>
Розовая  - тренировочные данные
### epoch categorical accuracy
![](./graphic/epoch_categorical_accuracy(2).svg)
### epoch loss
![](./graphic/epoch_loss(2).svg)
***
## Анализ


# 2. Transfer Learning
## Архитектура 
```python 
  inputs = tf.keras.layers.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
  model.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(model.output)     
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
````
***
## Визуализация обучения
Оранжевая - данные для валидации(проверки качества) <br/>
Синяя  - тренировочные данные
### epoch categorical accuracy
![](./graphic/epoch_categorical_accuracy(4).svg)
### epoch loss
![](./graphic/epoch_loss(4).svg)
# Анализ
Из графиков мы видим что точность на валидационном наборе данных достигает 0.95 <br/>
Отмечаем также быструю сходимость
