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

## SSE
Реализована технология SSE для фильтров Собеля и вычисления модулей градиента

    Naive implementation (no SSE) : Time [0.614] seconds
    SSE implementation : Time [0.561] seconds

    Ubuntu 16.04
    System: Intel Core i5-6200U 2.30GHz, 8Gb

## Сборка

    make all
    make clean # очистка

### Запуск из ./build/bin
обучение классификатора

    ./task2 -d ../../data/multiclass/train_labels.txt -m model.txt --train

классификация

    ./task2 -d ../../data/multiclass/test_labels.txt -m model.txt -l predictions.txt --predict

проверка точности классификации

    ./compare.py data/multiclass/test_labels.txt build/bin/predictions.txt

### Тестовые скрипты 
Сборка и запуск программы
    
    bash run.sh

Тестирование с помощью утилиты Google Test. Больше информации ./tests/readme.md

    bash runTest.sh

Документирование с помощью Doxygen

    bash runDocs.sh

Очистка проекта. Возвращение проекту исходный вид

    bash runClean.sh