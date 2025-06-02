import cv2
import torch
import numpy as np
from pathlib import Path
from groundingdino.util.inference import Model, predict
from typing import List, Tuple

class VideoPeopleDetector:
    """
    Детектор людей в видео с использованием Grounding DINO.
    
    Attributes:
        model (Model): Модель Grounding DINO
        device (str): Устройство для вычислений (cuda или cpu)
        BOX_THRESHOLD (float): Порог для bounding boxes
        TEXT_THRESHOLD (float): Порог для текстовых описаний
    """
    def __init__(self, config_path: str, weights_path: str):
        """
        Инициализация детектора.
        
        Args:
            config_path: Путь к конфигурационному файлу модели
            weights_path: Путь к файлу с весами модели
        """
        self.device = "cpu"
        self.model = Model(
            model_config_path=config_path,
            model_checkpoint_path=weights_path,
            device=self.device
        )
        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25
    
    def detect_people(self, frame: np.ndarray) -> Tuple[List[List[int]], List[float]]:
        """
        Детекция людей в кадре.
        
        Args:
            frame: Входной кадр в формате BGR
            
        Returns:
            Кортеж (boxes, scores):
                boxes: Список bounding boxes [x1, y1, x2, y2]
                scores: Список уверенностей для каждого box
        """
        # Преобразование кадра в RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Детекция объектов
        detections = self.model.predict_with_caption(
            image=image,
            caption="person",
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD
        )
        # Фильтрация и форматирование результатов
        
        boxes = []
        scores = []
        for box, score in zip(detections[0].xyxy, detections[0].confidence):
            boxes.append([int(x) for x in box])
            scores.append(float(score))
            
        return boxes, scores
    
    def process_video(self, input_path: str, output_path: str, skip_frames: int = 0):
        """
        Обработка видеофайла с детекцией людей.
        
        Args:
            input_path: Путь к входному видео
            output_path: Путь для сохранения результата
        """
        # Создание директорий при необходимости
        Path(input_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Открытие видеофайла
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл {input_path}")
        
        # Получение параметров видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создание VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Обработка каждого кадра
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1

            # Пропускаем кадры если нужно
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue
            
            print(f"Обработка кадра {frame_count}")
            
            # Детекция людей
            boxes, scores = self.detect_people(frame)
            
            # Отрисовка результатов
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Сохранение кадра
            out.write(frame)
        
        # Освобождение ресурсов
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Пути к конфигурации и весам
    CONFIG_PATH = "grounding-dino/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
    
    # Пути к видеофайлам
    INPUT_VIDEO = "input/crowd.mp4"
    OUTPUT_VIDEO = "output/crowd_detected.mp4"
    
    # Инициализация и запуск обработки
    print("Инициализация Grounding DINO...")
    detector = VideoPeopleDetector(CONFIG_PATH, WEIGHTS_PATH)
    
    print("Начало обработки видео...")
    detector.process_video(INPUT_VIDEO, OUTPUT_VIDEO)
    print(f"Обработанное видео сохранено в {OUTPUT_VIDEO}")