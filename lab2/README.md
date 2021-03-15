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


![](./graphic/epoch_categorical_accuracy(1).svg)
![](./graphic/epoch_loss(1).svg)
![](./graphic/epoch_categorical_accuracy(2).svg)
![](./graphic/epoch_loss(2).svg)
