import cv2
import numpy as np
import os
import glob
from pathlib import Path

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    print("ONNX Runtime не установлен. Установите: pip install onnxruntime")
    ONNX_AVAILABLE = False


class ONNXCarDetector:
    def __init__(self, model_size='s', confidence_threshold=0.5, test_image_dir='test_image'):
        """
        Инициализация ONNX детектора автомобилей
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime не доступен")

        self.confidence_threshold = confidence_threshold
        self.test_image_dir = test_image_dir

        # Путь к ONNX модели
        self.model_path = f'yolov8{model_size}.onnx'

        # Проверяем наличие модели
        if not os.path.exists(self.model_path):
            print(f"ONNX модель {self.model_path} не найдена!")
            print("Пытаемся загрузить и конвертировать PyTorch модель...")
            self._convert_pytorch_to_onnx(model_size)

        # Инициализируем ONNX Runtime
        try:
            # Используем CPU провайдер для совместимости
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            print(f"ONNX модель {self.model_path} успешно загружена")
        except Exception as e:
            print(f"Ошибка загрузки ONNX модели: {e}")
            raise

        # Получаем информацию о входных и выходных тензорах
        self.input_details = self.session.get_inputs()[0]
        self.output_details = self.session.get_outputs()[0]

        print(f"Вход: {self.input_details.name}, форма: {self.input_details.shape}")
        print(f"Выход: {self.output_details.name}, форма: {self.output_details.shape}")

        # Классы транспортных средств в COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        # Создаем каталог для результатов
        self.results_dir = os.path.join(test_image_dir, 'detection_results')
        os.makedirs(self.results_dir, exist_ok=True)

    def _convert_pytorch_to_onnx(self, model_size):
        """Конвертация PyTorch модели в ONNX"""
        try:
            from ultralytics import YOLO
            print("Загружаем PyTorch модель для конвертации...")

            model = YOLO(f'yolov8{model_size}.pt')
            print("Конвертируем в ONNX...")
            model.export(format='onnx', imgsz=640, simplify=True)
            print(f"Модель сконвертирована в {self.model_path}")

        except ImportError:
            print("Ultralytics не доступен для конвертации")
            raise FileNotFoundError(f"ONNX модель {self.model_path} не найдена и не может быть создана")

    def preprocess_image(self, image):
        """Предобработка изображения для ONNX модели"""
        # Сохраняем оригинальные размеры
        original_height, original_width = image.shape[:2]

        # Изменяем размер до 640x640 с сохранением пропорций
        target_size = 640
        scale = min(target_size / original_width, target_size / original_height)

        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Масштабируем изображение
        resized = cv2.resize(image, (new_width, new_height))

        # Создаем изображение 640x640 с паддингом
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Центрируем изображение
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2

        padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        # Нормализуем и меняем формат для модели
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Добавляем batch dimension

        return input_tensor, scale, x_offset, y_offset

    def postprocess_detections(self, output, scale, x_offset, y_offset, original_shape):
        """Постобработка результатов детекции"""
        detections = []

        # ONNX YOLOv8 выход имеет форму [1, 84, 8400] или [1, 84, num_detections]
        # где 84 = 4 координаты + 80 классов COCO
        predictions = output[0]  # Убираем batch dimension

        # Транспонируем для удобства: [num_detections, 84]
        if predictions.shape[0] == 84:
            predictions = predictions.T

        #print(f"Форма предсказаний после обработки: {predictions.shape}")

        for detection in predictions:
            # Извлекаем координаты центра и размеры
            x_center, y_center, width, height = detection[:4]

            # Извлекаем вероятности классов
            class_probs = detection[4:]

            # Находим класс с максимальной вероятностью
            max_class_id = np.argmax(class_probs)
            max_class_prob = class_probs[max_class_id]

            # Проверяем порог уверенности и принадлежность к транспортным средствам
            if max_class_prob > self.confidence_threshold and max_class_id in self.vehicle_classes:
                # Конвертируем координаты обратно в оригинальное изображение
                # Сначала убираем паддинг
                x_center = (x_center - x_offset) / scale
                y_center = (y_center - y_offset) / scale
                width = width / scale
                height = height / scale

                # Конвертируем в координаты углов
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # Ограничиваем координаты размерами изображения
                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))

                detection_dict = {
                    'class': self.class_names[max_class_id],
                    'confidence': float(max_class_prob),
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                }
                detections.append(detection_dict)

        return detections

    def detect_vehicles(self, frame, circle_center=None, circle_radius=None):
        """
        Детекция транспортных средств на кадре с опциональной проверкой круга
        """
        original_shape = frame.shape[:2]

        # Предобработка
        input_tensor, scale, x_offset, y_offset = self.preprocess_image(frame)

        # Инференс
        try:
            output = self.session.run([self.output_details.name], {self.input_details.name: input_tensor})
            #print(f"Выход ONNX модели: {[o.shape for o in output]}")
        except Exception as e:
            print(f"Ошибка при выполнении ONNX инференса: {e}")
            return frame, []

        # Постобработка
        detections = self.postprocess_detections(output[0], scale, x_offset, y_offset, original_shape)

        # Аннотация кадра
        annotated_frame = self._annotate_frame(frame, detections, circle_center, circle_radius)

        return annotated_frame, detections

    def detect_vehicles_in_circle(self, frame, center_point, radius, detection_mode='circle_bbox_intersect'):
        """
        Детекция транспортных средств с проверкой пересечения круга с bounding box
        """
        # Сначала получаем все детекции
        annotated_frame, all_detections = self.detect_vehicles(frame, center_point, radius)

        # Фильтруем детекции в круге
        detections_in_circle = []

        circle_center_x = int(center_point[0])
        circle_center_y = int(center_point[1])
        circle_radius = int(radius)

        for detection in all_detections:
            x1, y1, x2, y2 = detection['bbox']

            # Проверяем пересечение с кругом
            is_intersecting = self._check_circle_bbox_intersection(
                circle_center_x, circle_center_y, circle_radius,
                x1, y1, x2, y2
            )

            if is_intersecting:
                detections_in_circle.append(detection)

        # Перерисовываем аннотации с правильными цветами
        annotated_frame = self._annotate_frame_with_circle(
            frame, all_detections, detections_in_circle, center_point, radius
        )

        return annotated_frame, detections_in_circle

    def _annotate_frame(self, frame, detections, circle_center=None, circle_radius=None):
        """Аннотация кадра с детекциями"""
        annotated_frame = frame.copy()

        # Рисуем круг если задан
        if circle_center is not None and circle_radius is not None:
            cv2.circle(annotated_frame, (int(circle_center[0]), int(circle_center[1])),
                       int(circle_radius), (255, 255, 0), 3)  # Голубой круг
            cv2.circle(annotated_frame, (int(circle_center[0]), int(circle_center[1])),
                       10, (0, 0, 255), -1)  # Красная центральная точка

        # Рисуем детекции
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']

            # Цвет рамки
            color = (0, 255, 0)  # Зеленый по умолчанию

            # Рисуем прямоугольник
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Подпись
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Фон для текста
            cv2.rectangle(annotated_frame,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1),
                          color, -1)

            # Текст
            cv2.putText(annotated_frame, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # Центральная точка
            center_x, center_y = detection['center']
            cv2.circle(annotated_frame, (center_x, center_y), 5, color, -1)

        return annotated_frame

    def _annotate_frame_with_circle(self, frame, all_detections, detections_in_circle, center_point, radius):
        """Аннотация кадра с различными цветами для детекций в круге и вне его"""
        annotated_frame = frame.copy()

        # Рисуем круг
        cv2.circle(annotated_frame, (int(center_point[0]), int(center_point[1])),
                   int(radius), (255, 255, 0), 3)  # Голубой круг
        cv2.circle(annotated_frame, (int(center_point[0]), int(center_point[1])),
                   10, (0, 0, 255), -1)  # Красная центральная точка

        # Создаем множество детекций в круге для быстрого поиска
        in_circle_centers = {det['center'] for det in detections_in_circle}

        # Рисуем все детекции с соответствующими цветами
        for detection in all_detections:
            x1, y1, x2, y2 = detection['bbox']

            # Определяем цвет в зависимости от того, в круге ли детекция
            if detection['center'] in in_circle_centers:
                color = (0, 255, 0)  # Зеленый для детекций в круге
            else:
                color = (0, 0, 255)  # Красный для детекций вне круга

            # Рисуем прямоугольник
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Подпись
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Фон для текста
            cv2.rectangle(annotated_frame,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1),
                          color, -1)

            # Текст
            cv2.putText(annotated_frame, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # Центральная точка
            center_x, center_y = detection['center']
            cv2.circle(annotated_frame, (center_x, center_y), 5, color, -1)

        return annotated_frame

    def _check_circle_bbox_intersection(self, circle_x, circle_y, circle_radius, bbox_x1, bbox_y1, bbox_x2, bbox_y2):
        """Проверка пересечения круга с прямоугольником"""
        # Находим ближайшую точку прямоугольника к центру круга
        closest_x = max(bbox_x1, min(circle_x, bbox_x2))
        closest_y = max(bbox_y1, min(circle_y, bbox_y2))

        # Вычисляем расстояние
        distance_x = circle_x - closest_x
        distance_y = circle_y - closest_y
        distance_squared = distance_x * distance_x + distance_y * distance_y

        return distance_squared <= (circle_radius * circle_radius)

    def count_vehicles_by_type(self, detections):
        """Подсчет транспортных средств по типам"""
        counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'total': 0}

        for detection in detections:
            vehicle_type = detection['class']
            if vehicle_type in counts:
                counts[vehicle_type] += 1
                counts['total'] += 1

        return counts

    # Остальные методы для совместимости с оригинальным API
    def get_image_files(self):
        """Получение списка файлов изображений"""
        if not os.path.exists(self.test_image_dir):
            return []

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []

        for extension in image_extensions:
            pattern = os.path.join(self.test_image_dir, extension)
            image_files.extend(glob.glob(pattern))
            pattern_upper = os.path.join(self.test_image_dir, extension.upper())
            image_files.extend(glob.glob(pattern_upper))

        return sorted(image_files)

    def analyze_test_images(self, save_results=True):
        """Анализ изображений из каталога test_image"""
        print(f"Анализ изображений ONNX детектором из: {self.test_image_dir}")

        image_files = self.get_image_files()
        if not image_files:
            print("Изображения не найдены!")
            return []

        results = []

        for image_path in image_files:
            print(f"Обрабатываем: {os.path.basename(image_path)}")

            frame = cv2.imread(image_path)
            if frame is None:
                continue

            annotated_frame, detections = self.detect_vehicles(frame)
            counts = self.count_vehicles_by_type(detections)

            result = {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'detections': detections,
                'counts': counts,
                'total_vehicles': counts['total']
            }

            results.append(result)
            print(f"  Найдено ТС: {counts['total']}")

            if save_results and counts['total'] > 0:
                # Сохраняем результат
                name_without_ext = os.path.splitext(result['image_name'])[0]
                result_filename = f"{name_without_ext}_onnx_detected.jpg"
                result_path = os.path.join(self.results_dir, result_filename)
                cv2.imwrite(result_path, annotated_frame)

        return results


# Функция для тестирования
def test_onnx_detector():
    """Тестирование ONNX детектора"""
    try:
        detector = ONNXCarDetector(model_size='s', confidence_threshold=0.5)
        print("ONNX детектор успешно инициализирован!")

        # Тест на пустом изображении
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, detections = detector.detect_vehicles(test_image)
        print(f"Тест пройден. Найдено детекций: {len(detections)}")

        return True

    except Exception as e:
        print(f"Ошибка тестирования ONNX детектора: {e}")
        return False


if __name__ == "__main__":
    test_onnx_detector()