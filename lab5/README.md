# Решение задачи классификации изображений из набора данных Oregon Wildlife с использованием нейронных сетей глубокого обучения и техники обучения Fine Tuning
## 1. Тренировка без применения Fine Tuning
![iXQQxiSlZuI](https://user-images.githubusercontent.com/61012068/113757677-9256b800-971b-11eb-8ea9-88c50275360b.jpg) </br>
accuracy
![](./graphic/before_accuracy.svg)
loss
![](./graphic/before_loss.svg)

## 2. Нахождение оптимального темпа обучения 
### 2.1 Статический 
Ниже представлены графики обучения с ```lr = 1e-7, 1e-8, 1e-9, 1e-10``` </br></br>
![wbVQ9f-m1R4](https://user-images.githubusercontent.com/61012068/113757708-9c78b680-971b-11eb-9f9f-22f164545b64.jpg) </br>
accuracy
![](./graphic/lrs_accuracy.svg)
loss
![](./graphic/lrs_loss.svg)
Из 4-х вариантов у ```lr = 1e-9``` точность в среднем  на ```~0.007``` больше чем у ```lr = 1e-8``` и у ```lr = 1e-10```. Его у будем считать оптимальным из предложенных вариантов </br>

### 2.2 Изменяющийся по экспоненциальному закону
Формула изменения темпа обучния имеет следующий вид:
```python
lrate = 1e-8 * exp(-k * num_epoch)
``` 
Где ```k = 0.3, 0.5, 0.7, 0.9``` </br></br>
![YDhkuCh4haM](https://user-images.githubusercontent.com/61012068/113771611-452f1200-972c-11eb-815e-8d0c6f23eec6.jpg) </br>
accuracy
![](./graphic/exp_accuracy.svg)
loss
![](./graphic/exp_loss.svg)
Среди рассмотренных вариантов очень трудно выделить оптимальный так как разница в точности с среднем ```~0.002```. Для средующих сравнений выберем вариант с ```k = 0.9```.
### 2.3 Изменяющийся по ступенчатому закону 
```python
lrate = 1e-8 * drop^floor(epoch / epoch_drop) 
```
Где ```drop = 0.5, 0.75, 0.98```, ```epoch_drop = 10, 75, 0.98``` соответственно  </br></br>
![n-uos1f0RqM](https://user-images.githubusercontent.com/61012068/113762092-e44e0c80-9720-11eb-8295-b679c3bb1310.jpg) </br>
accuracy
![](./graphic/step_accuracy.svg)
loss
![](./graphic/step_loss.svg)
У варианта с ```drop = 0.5, epoch_drop = 10``` точность в среднем на ```~0.006``` больше чем у других, его и будем считать оптимальным их данных 3-х вариантов.
### 2.4 Сравнение 3-х вышеописанных способов инициализации темпа обучения 
![n1misDfq4Xw](https://user-images.githubusercontent.com/61012068/113769032-3004b400-9729-11eb-97ef-c27244a32331.jpg) </br>
accuracy
![](./graphic/all_accuracy.svg)
loss
![](./graphic/all_loss.svg)
Среди 3-х представленных вариантов лучше всего показал себя ```lr = 1e-9```. Точность больше в среднем на ```~0.008```, ошибка меньше в среднем  на ```0.0067```
## 3. Тренировка с применением Fine Tuning
![99u4N7chvRk](https://user-images.githubusercontent.com/61012068/113769335-8f62c400-9729-11eb-85f5-b40dbb76cdbb.jpg) </br>
accuracy
![](./graphic/last_accuracy.svg)
loss
![](./graphic/last_loss.svg)
Применяя технику Fine Tuning были достигнуты следующие резьтаты: точность увеличилась в среднем на ```~0.0038```, ошибка уменьшилась в среднем на ```0.0078```
