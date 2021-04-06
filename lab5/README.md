# Решение задачи классификации изображений из набора данных Oregon Wildlife с использованием нейронных сетей глубокого обучения и техники обучения Fine Tuning
## 1. Тренировка без применения Fine Tuning
![iXQQxiSlZuI](https://user-images.githubusercontent.com/61012068/113757677-9256b800-971b-11eb-8ea9-88c50275360b.jpg)
accuracy
![](./graphic/before_accuracy.svg)
loss
![](./graphic/before_loss.svg)

## 2. Нахождение оптимального темпа обучения 
### 1 Статический 
![wbVQ9f-m1R4](https://user-images.githubusercontent.com/61012068/113757708-9c78b680-971b-11eb-9f9f-22f164545b64.jpg)
accuracy
![](./graphic/lrs_accuracy.svg)
loss
![](./graphic/lrs_loss.svg)


![VoYZ2IrPXqY](https://user-images.githubusercontent.com/61012068/113761691-6b4eb500-9720-11eb-82b9-d4c8d736f4f1.jpg)
accuracy
![](./graphic/e-8_accuracy.svg)
loss
![](./graphic/e-8_loss.svg)

### 2. Изменяющийся по экспоненциальному закону
![ry7RNiX28sA](https://user-images.githubusercontent.com/61012068/113761729-79043a80-9720-11eb-9197-c950155635bb.jpg)
accuracy
![](./graphic/exp_accuracy.svg)
loss
![](./graphic/exp_loss.svg)

### 3. Изменяющийся по ступенчатому закону 
![n-uos1f0RqM](https://user-images.githubusercontent.com/61012068/113762092-e44e0c80-9720-11eb-8295-b679c3bb1310.jpg)
accuracy
![](./graphic/step_accuracy.svg)
loss
![](./graphic/step_loss.svg)

### Сравнение 3-х вышеописанных способов инициализации темпа обучения 
![STNt42kADwU](https://user-images.githubusercontent.com/61012068/113762198-00ea4480-9721-11eb-9789-2da695ec980c.jpg)
accuracy
![](./graphic/all_accuracy.svg)
loss
![](./graphic/all_loss.svg)

## 3. Тренировка с применением Fine Tuning
![URq-AFEk0MM](https://user-images.githubusercontent.com/61012068/113757766-a995a580-971b-11eb-95a2-1765a98f9fe0.jpg)
accuracy
![](./graphic/last_accuracy.svg)
loss
![](./graphic/last_loss.svg)
