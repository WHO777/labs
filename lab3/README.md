# Изучение влияние параметра “темп обучения” на процесс обучения нейронной сети на примере решения задачи классификации Oregon Wildlife с использованием техники обучения Transfer Learning
## ```# static_lr.py```
## 1. Графики тренировки нейронной сети со статическим темпом обучения 

### epoch categorical accuracy
![9EpJfjgo2as](https://user-images.githubusercontent.com/61012068/111904289-302d6000-8a57-11eb-8238-659a7749af1a.jpg)
![](./graphic/static_categorical_accuracy.svg)
### epoch loss
![](./graphic/static_loss.svg)
***
## Анализ

## ```# n_static_lr.py```

### Формула
```python
lrate = initial_lrate * exp(-k * num_epoch)
```
## 2. Графики тренировки нейронной сети с темпом обучения изменяющегося по экспоненциальному закону с параметрами: </br>
```initial_lrate = 0.1``` </br> 
```k = 0.1, 0,2, ..., 0.5``` </br> 
```num_epoch = 1, 2, ..., 50``` </br> </br>
![изображение](https://user-images.githubusercontent.com/61012068/111904308-505d1f00-8a57-11eb-92b4-b09483f01d86.png)

### epoch categorical accuracy
![](./graphic/exp_categorical_accuracy.svg)
### epoch loss
![](./graphic/exp_loss.svg)
***
## Анализ

### Формула
```python
lrate = initial_lrate * drop^floor(epoch / epochs_drop) 
```
## 3. Графики тренировки нейронной сети с темпом обучения изменяющегося по "ступенчатому" закону с параметрами: </br>
```initial_lrate = 0.1``` </br> 
```drop = 0.99, 0.95, 0.5, 0.4, 0.35, 0.3, 0.1``` </br> 
```epochs_drop = 1, 2, 10, 7, 7, 5, 1``` </br> </br> 
![изображение](https://user-images.githubusercontent.com/61012068/111904315-59e68700-8a57-11eb-9088-8b8d958053a3.png)

### epoch categorical accuracy
![](./graphic/step_categorical_accuracy.svg)
### epoch loss
![](./graphic/step_loss.svg)
***
# Анализ
