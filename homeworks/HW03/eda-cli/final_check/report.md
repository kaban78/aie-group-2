# EDA Report

*Сгенерировано: 2025-12-05 22:41:20*


## Параметры анализа
- Исходный файл: `example.csv`
- Разделитель: `,`
- Кодировка: `utf-8`
- Максимум гистограмм: `6`
- Топ категорий: `3`
- Порог пропусков: `0.3`
- Проверка выбросов: `Нет`

## Основная информация
- Строк: **36**
- Колонок: **14**

## Качество данных
- Общий скор качества: **0.74**
- Too Few Rows: **True**
- Too Many Columns: **False**
- Too Many Missing: **False**

### Новые эвристики качества (HW03)
- Has Constant Columns: ❌ Нет
- Has Imbalanced Categories: ❌ Нет
- Has High Cardinality Categoricals: ❌ Нет
- Has Suspicious Id Duplicates: ❌ Нет

## Пропуски
Таблица пропусков сохранена в `missing.csv`
Визуализация пропусков: `missing_matrix.png`


## Корреляция числовых признаков
Корреляционная матрица сохранена в `correlation.csv`
Тепловая карта корреляций: `correlation_heatmap.png`


## Категориальные признаки
Топ-3 категорий:

### country
- `RU`: 21 (67.7%)
- `KZ`: 5 (16.1%)
- `BY`: 5 (16.1%)

### city
- `Moscow`: 11 (64.7%)
- `Saint Petersburg`: 3 (17.6%)
- `Almaty`: 3 (17.6%)

### device
- `Desktop`: 17 (47.2%)
- `Mobile`: 15 (41.7%)
- `Tablet`: 4 (11.1%)

### channel
- `Organic`: 16 (53.3%)
- `Ads`: 8 (26.7%)
- `Referral`: 6 (20.0%)

### plan
- `Basic`: 14 (38.9%)
- `Free`: 12 (33.3%)
- `Pro`: 10 (27.8%)

## Визуализации
- Гистограммы числовых колонок (максимум 6): `hist_*.png`
- Матрица пропусков: `missing_matrix.png`
- Тепловая карта корреляций: `correlation_heatmap.png`