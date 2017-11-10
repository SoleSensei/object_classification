# Object Classification
Машграф #2 Классификация объектов + SSE

## Базовая часть
Гистограммы ориентированных градиентов

    1) grayscale
    2) свертка фильтром Собеля
    3) градиенты яркости пикселей
    4) гистограммы градиентов + нормализация
    5) конкатенация гистограмм в дескриптор

## Дополнительная часть
Локальные бинарные шаблоны
Цветовые признаки

## Сборка

    make all
    make clean - очистка

### Запуск из ./build/bin
обучение классификатора

    ./task2 -d ../../data/multiclass/train_labels.txt -m model.txt --train

классификация

    ./task2 -d ../../data/multiclass/test_labels.txt -m model.txt -l predictions.txt --predict

проверка точности классификации

    ./compare.py data/multiclass/test_labels.txt build/bin/predictions.txt

### Тестовый скрипт test.sh

    bash test.sh