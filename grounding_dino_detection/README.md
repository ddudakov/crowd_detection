# Детекция людей на видео с Grounding DINO

Проект для детекции людей с использованием zero-shot модели Grounding DINO.
Ссылка на гит - https://github.com/IDEA-Research/GroundingDINO/tree/main

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/people-detection-groundingdino.git
cd people-detection-groundingdino

2. Установите зависимости:
```bash
pip install -r requirements.txt

3. Скачать веса для Grounding DINO:
```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

И переместить их в фолдер *weights*:
```bash
mv groundingdino_swint_ogc.pth grounding_dino_detection/weights/

## Использование
```bash
python main.py

Результат будет сохранен в *output/crowd_detected.mp4*

## Особенности

1. Zero-shot детекция (не требует обучения);

2. Возможность детекции по текстовым описаниям;

3. Хорошая точность на нестандартных объектах;

4. Поддержка нескольких классов одновременно;

## Ограничения

1. Высокие требования к памяти

2. Медленнее YOLO