# EN 

## CIFAR10 Image Classification Problem

A pipeline for CIFAR10 image classification based on the SimpleNet model using additional layers is presented.

Structure:
- notebooks:

     showing results:
     - start training the model (check_initial_training.ipynb);
     - model saving checks (check_resume_training);
     - compare cpu and gpu (compare_cpu_gpu).

     running the model training pipeline in a notebook using additional layers (CIFAR2.ipynb)

- src:
    
     The pipeline is decomposed into .py files, for learning to run initial_training.py


The dataset will be loaded into the input folder;
the weights and graphs of the loss function and accuracy are saved in the outputs folder.

# RU

## Задача классификации изображений CIFAR10

Представлен пайплаин для классификации изображений CIFAR10 на основе модели SimpleNet с использованием дополнительных слоёв.

Структура:
- notebookes:

    показаны результаты:
    - запуска обучения модели (check_initial_training.ipynb);
    - проверки сохранения модели (check_resume_training);
    - сравнения cpu и gpu (compare_cpu_gpu).

    запуск пайплайна обучения модели в ноутбуке с использованием дополнительных слоёв (CIFAR2.ipynb)

- src:
    
    пайплаин декомпозирован по файлам .py, для обучения запускать initial_training.py


Загрузка датасета произойдет в папку input;
сохранение весов и графиков функции потерь и точности происходит в папку outputs.