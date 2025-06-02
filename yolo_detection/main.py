import cv2
from ultralytics import YOLO
from pathlib import Path

def process_video(input_path: str, output_path: str, model_name: str = 'yolov8n.pt'):
    """
    Обрабатывает видеофайл, детектируя людей и сохраняя результат с bounding boxes.
    
    Args:
        input_path (str): Путь к входному видеофайлу
        output_path (str): Путь для сохранения обработанного видео
        model_name (str): Название модели YOLO (по умолчанию 'yolov8n.pt')
    """
    # Создаем папки, если они не существуют
    Path(input_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Загружаем модель
    model = YOLO(model_name)
    
    # Открываем видеофайл
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видеофайл {input_path}")
    
    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Создаем VideoWriter для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Обрабатываем каждый кадр
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детекция объектов
        results = model(frame, verbose=False)
        
        # Отрисовка bounding boxes
        for result in results:
            for box in result.boxes:
                # Проверяем, что это человек (класс 0 в YOLO)
                if int(box.cls) == 0:
                    # Получаем координаты и уверенность
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Отрисовываем прямоугольник
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Добавляем текст с классом и уверенностью
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Сохраняем кадр
        out.write(frame)
    
    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = "input/crowd.mp4"
    output_video = "output/crowd_detected.mp4"
    
    print("Начало обработки видео...")
    process_video(input_video, output_video)
    print(f"Обработанное видео сохранено в {output_video}")