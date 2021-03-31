# Использование техник аугментации данных для улучшения сходимости процесса обучения нейронной сети на примере решения задачи классификации Oregon Wildlife
Ниже представлены результаты тренировки EfficientNet-B0 (предварительно обученной на базе изображений imagenet) с использованием 4-ех видов аугментации, а таксе их композиция. 
## 1. Манипуляции с яркостью и контрастом
### ```# brightness_contrast.py```

Изначально формула для темпа обучения использовалась следующая:
1. ```python
      lrate = 0.01 * exp(-0.3 * num_epoch)
   ```
После 3-ех тренировок она была заменена на:
2. ```python
      lrate = 0.01 * exp(-0.3 * num_epoch)
   ```
Сраввнивая эти два способа изменения темпа обучения видно что второй показал себя лучше и поэтому будет использваться далее везде, где не будет уточнено обратное
На легенде ниже у графиков полученных с помощью первого способа задания темпа обучения в имени отсутствует параметр ```k```. Остальные графики были получены с импользванием второго способа.
![FpjL-Y_lbmQ](https://user-images.githubusercontent.com/61012068/113120679-826e3e00-921a-11eb-8ae3-c651e139a5e3.jpg)

![](./graphic/BrightnessContrast_accuracy.svg)
![](./graphic/BrightnessContrast_loss.svg)

## 2. Поворот изображения на случайный угол
### ```# rotate.py```
![c8dky2M7uAY](https://user-images.githubusercontent.com/61012068/113120713-89954c00-921a-11eb-8c53-cc573e9b2a2a.jpg)

![](./graphic/Rotate_accuracy.svg)
![](./graphic/Rotate_loss.svg)

## 3. Использование случайной части изображения
### ```# crop.py```
![PyaxZp33zNQ](https://user-images.githubusercontent.com/61012068/113120720-8c903c80-921a-11eb-95b6-ab515b509f4d.jpg)

![](./graphic/RandomCrop_accuracy.svg)
![](./graphic/RandomCrop_loss.svg)

## 4. Добавление случайного шума
### ```# noise.py```
![tYMn-ArmZmw](https://user-images.githubusercontent.com/61012068/113120731-8f8b2d00-921a-11eb-8aee-9e6810aa77c4.jpg)

![](./graphic/GaussNoise_accuracy.svg)
![](./graphic/GaussNoise_loss.svg)

## 5. Композиция
```python
print('fd')
```
