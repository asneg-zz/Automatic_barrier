import sys
import cv2
import numpy as np
import time
import os
import threading
import queue
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QFileDialog, QSpinBox,
                             QDoubleSpinBox, QTextEdit, QGroupBox, QGridLayout,
                             QComboBox, QProgressBar, QSlider, QCheckBox)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, Qt, QMutex, QMutexLocker, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPolygon

# Импорт UI layout из отдельного файла
from ui_layout import init_ui

# Импорт YOLO детектора
try:
    from onnx_car_detector_fixed import ONNXCarDetector as YOLOCarDetector

    YOLO_AVAILABLE = True
except ImportError:
    print("Предупреждение: YOLOCarDetector не найден. Функция детекции автомобилей будет недоступна.")
    YOLO_AVAILABLE = False


class VideoLabel(QLabel):
    """Кастомный QLabel для отображения видео с возможностью рисования полигона"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = None
        self.polygon_points = []
        self.drawing_polygon = False
        self.scale_x = 1.0
        self.scale_y = 1.0

    def set_parent_window(self, parent_window):
        """Установка родительского окна"""
        self.parent_window = parent_window

    def mousePressEvent(self, event):
        """Обработка кликов мыши с отладкой"""
        if self.parent_window and self.drawing_polygon:

            # Получаем координаты клика относительно ИСХОДНОГО изображения
            x, y = self.get_image_coordinates(event.x(), event.y())

            if event.button() == Qt.LeftButton:
                # Добавляем точку полигона
                self.polygon_points.append((x, y))
                self.parent_window.log_message(f"Добавлена точка полигона: ({x}, {y})")
                self.update()  # Перерисовываем виджет

            elif event.button() == Qt.RightButton:
                # Завершаем рисование полигона (круга)
                if len(self.polygon_points) >= 2:
                    # Создаем круговой полигон из двух точек (центр и радиус)
                    center_point = self.polygon_points[0]
                    edge_point = self.polygon_points[-1]

                    # Вычисляем радиус
                    radius = ((edge_point[0] - center_point[0]) ** 2 +
                              (edge_point[1] - center_point[1]) ** 2) ** 0.5

                    print("ris", center_point, radius)

                    # Создаем точки круга (аппроксимируем полигоном)
                    import math
                    circle_points = []
                    num_points = 32  # Количество точек для аппроксимации круга
                    for i in range(num_points):
                        angle = 2 * math.pi * i / num_points
                        cx = center_point[0] + radius * math.cos(angle)
                        cy = center_point[1] + radius * math.sin(angle)
                        circle_points.append((int(cx), int(cy)))

                    # Устанавливаем параметры круга в главном окне
                    self.parent_window.polygon_points = circle_points
                    self.parent_window.circle_center = center_point
                    self.parent_window.circle_radius = radius

                    # Сбрасываем флаги рисования
                    self.drawing_polygon = False
                    self.parent_window.drawing_polygon = False

                    self.parent_window.log_message(f"Круг создан с центром {center_point} и радиусом {radius:.1f}")

                else:
                    self.parent_window.log_message("Ошибка: Для круга нужно задать центр и точку на окружности")

                self.update()

    def widget_to_image_coordinates(self, widget_x, widget_y):
        """Преобразование координат виджета в координаты изображения"""
        return self.get_image_coordinates(widget_x, widget_y)

    def image_to_widget_coordinates(self, image_x, image_y):
        """Преобразование координат изображения в координаты виджета"""
        if self.pixmap() is None:
            return image_x, image_y

        widget_size = self.size()
        pixmap_size = self.pixmap().size()

        # Вычисляем масштаб
        scale_x = pixmap_size.width() / widget_size.width()
        scale_y = pixmap_size.height() / widget_size.height()
        scale = max(scale_x, scale_y)

        # Вычисляем смещение
        scaled_width = pixmap_size.width() / scale
        scaled_height = pixmap_size.height() / scale
        offset_x = (widget_size.width() - scaled_width) / 2
        offset_y = (widget_size.height() - scaled_height) / 2

        # Преобразуем координаты
        widget_x = int(image_x / scale + offset_x)
        widget_y = int(image_y / scale + offset_y)

        return widget_x, widget_y

    def paintEvent(self, event):
        """Переопределение метода рисования для добавления полигона"""
        super().paintEvent(event)

        if len(self.polygon_points) > 0:
            painter = QPainter(self)
            self.draw_polygon_on_widget(painter)
            painter.end()

    def get_image_coordinates(self, widget_x, widget_y):
        """Преобразование координат виджета в координаты ОТОБРАЖАЕМОГО изображения (не оригинального)"""
        if self.pixmap() is None:
            return widget_x, widget_y

        # Получаем размеры виджета и пиксмапа (масштабированного изображения)
        widget_size = self.size()
        pixmap_size = self.pixmap().size()

        # Вычисляем масштаб между виджетом и отображаемым пиксмапом
        scale_x = pixmap_size.width() / widget_size.width()
        scale_y = pixmap_size.height() / widget_size.height()
        scale = max(scale_x, scale_y)

        # Вычисляем размеры масштабированного изображения в виджете
        scaled_width = pixmap_size.width() / scale
        scaled_height = pixmap_size.height() / scale

        # Вычисляем смещение изображения в виджете (центрирование)
        offset_x = (widget_size.width() - scaled_width) / 2
        offset_y = (widget_size.height() - scaled_height) / 2

        # Преобразуем координаты в координаты отображаемого изображения
        image_x = int((widget_x - offset_x) * scale)
        image_y = int((widget_y - offset_y) * scale)

        # Ограничиваем координаты размерами отображаемого изображения
        image_x = max(0, min(image_x, pixmap_size.width() - 1))
        image_y = max(0, min(image_y, pixmap_size.height() - 1))

        return image_x, image_y

    def draw_polygon_on_widget(self, painter):
        """Рисование полигона в виде круга на виджете средствами PyQt5"""
        if len(self.polygon_points) < 1:
            return

        # Получаем масштаб для правильного отображения
        if self.pixmap() is None:
            return

        if len(self.polygon_points) == 1:
            # Рисуем центральную точку
            center_point = self.polygon_points[0]
            widget_x, widget_y = self.image_to_widget_coordinates(center_point[0], center_point[1])

            painter.setPen(QPen(QColor(255, 0, 0), 2))  # Красная точка
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(widget_x - 5, widget_y - 5, 10, 10)

        elif len(self.polygon_points) >= 2:
            # Вычисляем центр и радиус круга
            center_point = self.polygon_points[0]
            edge_point = self.polygon_points[-1]  # Последняя точка определяет радиус

            # Преобразуем координаты в координаты виджета
            center_x, center_y = self.image_to_widget_coordinates(center_point[0], center_point[1])
            edge_x, edge_y = self.image_to_widget_coordinates(edge_point[0], edge_point[1])

            # Вычисляем радиус в координатах виджета
            radius = int(((edge_x - center_x) ** 2 + (edge_y - center_y) ** 2) ** 0.5)

            # Настройка пера для круга
            pen = QPen(QColor(0, 255, 255))  # Голубой цвет для круга
            pen.setWidth(3)
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 255, 255, 30))  # Полупрозрачная заливка

            # Рисуем круг
            painter.drawEllipse(center_x - radius, center_y - radius,
                                radius * 2, radius * 2)

            # Рисуем центральную точку
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(center_x - 5, center_y - 5, 10, 10)

            # Рисуем точку на краю (показывает радиус)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(QColor(0, 255, 0))
            painter.drawEllipse(edge_x - 3, edge_y - 3, 6, 6)


class VideoAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Инициализация YOLO детектора
        if YOLO_AVAILABLE:
            try:
                self.yolo_detector = YOLOCarDetector(
                    model_size='s',
                    confidence_threshold=0.5,
                    test_image_dir='test_image'
                )
                self.yolo_available = True
                print("YOLO детектор успешно инициализирован")
            except Exception as e:
                print(f"Ошибка инициализации YOLO: {e}")
                self.yolo_detector = None
                self.yolo_available = False
        else:
            self.yolo_detector = None
            self.yolo_available = False

        # Переменные состояния
        self.video_source = None
        self.polygon_points = []
        self.circle_center = None
        self.circle_radius = None
        self.drawing_polygon = False
        self.vehicle_count = 0
        self.is_running = False
        self.current_frame = None
        self.mutex = QMutex()

        # Видео захват
        self.cap = None
        self.capture_thread = None

        # Переменные для записи видео
        self.is_recording = False
        self.video_writer = None
        self.output_filename = None
        self.recording_mutex = QMutex()

        # Режимы работы
        self.vehicle_detection_enabled = True

        # Инициализация UI из внешнего модуля
        init_ui(self)

        # Проверяем доступность YOLO
        self._check_yolo_availability()

        # Заменяем обычный QLabel на кастомный VideoLabel
        self.setup_custom_video_label()

        # Таймер для обновления видео
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_display)
        self.timer.start(33)  # ~30 FPS

    def _check_yolo_availability(self):
        """Проверка и настройка доступности YOLO"""
        if not self.yolo_available:
            self.vehicle_detection_checkbox.setText("Детекция автомобилей (YOLO) - недоступно")
            self.vehicle_detection_checkbox.setEnabled(False)
            self.yolo_confidence_slider.setEnabled(False)

    def toggle_vehicle_detection(self, state):
        """Переключение детекции автомобилей"""
        if not self.yolo_available:
            self.log_message("YOLO детекция недоступна!")
            return

        self.vehicle_detection_enabled = state == Qt.Checked
        self.log_message(f"Детекция автомобилей: {'включена' if self.vehicle_detection_enabled else 'выключена'}")

    def update_yolo_confidence(self, value):
        """Обновление порога уверенности YOLO"""
        if self.yolo_detector:
            confidence = value / 100.0
            self.yolo_detector.confidence_threshold = confidence
            self.yolo_confidence_label.setText(f"{confidence:.2f}")

    def setup_custom_video_label(self):
        """Настройка кастомного video label"""
        # Найдем parent layout для video_label
        parent_widget = None
        parent_layout = None

        # Ищем video_label в дереве виджетов
        for widget in self.findChildren(QLabel):
            if widget == self.video_label:
                parent_widget = widget.parent()
                if parent_widget:
                    parent_layout = parent_widget.layout()
                break

        if parent_layout is None:
            # Если не найден через parent, ищем через central widget
            central_widget = self.centralWidget()
            main_layout = central_widget.layout()
            # Получаем левый layout (video layout)
            video_layout_item = main_layout.itemAt(0)
            if video_layout_item:
                parent_layout = video_layout_item.layout()

        if parent_layout:
            # Находим позицию старого label
            index = -1
            for i in range(parent_layout.count()):
                item = parent_layout.itemAt(i)
                if item and item.widget() == self.video_label:
                    index = i
                    break

            # Сохраняем параметры старого label
            old_minimum_size = self.video_label.minimumSize()
            old_style = self.video_label.styleSheet()
            old_text = self.video_label.text()
            old_alignment = self.video_label.alignment()

            # Удаляем старый label
            parent_layout.removeWidget(self.video_label)
            self.video_label.setParent(None)

            # Создаем новый кастомный label
            self.video_label = VideoLabel()
            self.video_label.set_parent_window(self)
            self.video_label.setMinimumSize(old_minimum_size)
            self.video_label.setStyleSheet(old_style)
            self.video_label.setAlignment(old_alignment)
            self.video_label.setText(old_text)

            # Вставляем новый label на ту же позицию
            if index >= 0:
                parent_layout.insertWidget(index, self.video_label, 1)  # stretch factor = 1
            else:
                parent_layout.addWidget(self.video_label, 1)

    def log_message(self, message):
        """Добавление сообщения в лог"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def select_video_file(self):
        """Выбор видео файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео файл", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*.*)"
        )
        if file_path:
            self.video_source = file_path
            self.log_message(f"Выбран файл: {file_path}")

    def select_camera(self):
        """Выбор камеры"""
        camera_id = self.camera_id_spin.value()
        self.video_source = camera_id
        self.log_message(f"Выбрана камера ID: {camera_id}")

    def start_recording(self):
        """Начало записи видео"""
        if not self.is_running:
            self.log_message("Ошибка: Запустите анализ видео перед записью!")
            return

        if self.is_recording:
            self.log_message("Ошибка: Запись уже идет!")
            return

        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filename = f"vehicle_detection_{timestamp}.mp4"

        # Создаем папку recordings если её нет
        recordings_dir = "recordings"
        os.makedirs(recordings_dir, exist_ok=True)

        output_path = os.path.join(recordings_dir, self.output_filename)

        # Настройка параметров записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0  # Частота кадров
        frame_size = (640, 480)  # Разрешение видео

        # Создаем объект VideoWriter
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        if not self.video_writer.isOpened():
            self.log_message("Ошибка: Не удалось создать файл для записи!")
            return

        # Устанавливаем флаг записи
        with QMutexLocker(self.recording_mutex):
            self.is_recording = True

        # Обновляем интерфейс
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(True)
        self.record_status_label.setText("Записывается...")

        self.log_message(f"Начата запись видео: {output_path}")
        self.log_message(f"Разрешение: {frame_size[0]}x{frame_size[1]}, FPS: {fps}")

    def stop_recording(self):
        """Остановка записи видео"""
        if not self.is_recording:
            self.log_message("Ошибка: Запись не ведется!")
            return

        # Сбрасываем флаг записи
        with QMutexLocker(self.recording_mutex):
            self.is_recording = False

        # Закрываем объект VideoWriter
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        # Обновляем интерфейс
        self.record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        self.record_status_label.setText("Не записывается")

        self.log_message(f"Запись остановлена. Файл сохранен: recordings/{self.output_filename}")

    def start_analysis(self):
        """Запуск анализа"""
        if self.video_source is None:
            self.log_message("Ошибка: Источник видео не выбран!")
            return

        if not self.vehicle_detection_enabled:
            self.log_message("Ошибка: Детекция автомобилей выключена!")
            return

        if self.is_running:
            self.log_message("Ошибка: Анализ уже работает!")
            return

        # Проверяем доступность источника перед запуском
        if isinstance(self.video_source, str):
            if not os.path.exists(self.video_source):
                self.log_message(f"Ошибка: Файл не найден - {self.video_source}")
                return

        self.is_running = True
        self.status_label.setText("Запускается...")

        # Запускаем поток захвата видео
        self.capture_thread = threading.Thread(
            target=self._video_capture_thread,
            args=(self.video_source,),
            daemon=True
        )
        self.capture_thread.start()

        self.log_message("Запуск анализа с детекцией автомобилей")

        # Даем время на инициализацию
        QTimer.singleShot(1000, self.check_video_status)

    def _video_capture_thread(self, source):
        """Поток захвата видео"""
        try:
            # Открываем источник видео
            if isinstance(source, str):
                self.cap = cv2.VideoCapture(source)
            else:
                self.cap = cv2.VideoCapture(int(source))

            if not self.cap.isOpened():
                print(f"Ошибка: Не удалось открыть источник видео: {source}")
                self.is_running = False
                return

            # Получаем информацию о исходном видео
            original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"Источник видео открыт успешно: {source}")
            print(f"Исходное разрешение: {original_width}x{original_height}")
            print(f"FPS: {fps}")
            print("Входное видео будет сжато до 640x480")

            # Основной цикл захвата
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Достигнут конец видео или ошибка чтения")
                    break

                # СЖИМАЕМ ВХОДНОЙ КАДР ДО 640x480
                compressed_frame = self.compress_frame_to_640x480(frame)

                # Сохраняем сжатый кадр с блокировкой
                with QMutexLocker(self.mutex):
                    self.current_frame = compressed_frame.copy()

                # Контроль частоты кадров (~30 FPS)
                time.sleep(0.033)

        except Exception as e:
            print(f"Ошибка в потоке захвата кадров: {e}")
        finally:
            if self.cap:
                self.cap.release()
                print("Источник видео закрыт")

    def check_video_status(self):
        """Проверка статуса видео после запуска"""
        if self.is_running:
            if self.cap and self.cap.isOpened():
                self.status_label.setText("Работает")
                self.log_message("Анализ успешно запущен")
            else:
                self.stop_analysis()
                self.status_label.setText("Ошибка")
                self.log_message("Ошибка: Не удалось открыть источник видео")

    def stop_analysis(self):
        """Остановка анализа"""
        self.is_running = False

        # Останавливаем запись если она идет
        if self.is_recording:
            self.stop_recording()

        # Ждем завершения потока
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        # Закрываем источник видео
        if self.cap:
            self.cap.release()
            self.cap = None

        self.status_label.setText("Остановлен")
        self.log_message("Анализ остановлен")

    def start_polygon_drawing(self):
        """Начало рисования полигона (круга) - С ОТЛАДКОЙ"""

        self.drawing_polygon = True
        self.video_label.drawing_polygon = True
        self.video_label.polygon_points = []
        self.polygon_points = []
        self.circle_center = None
        self.circle_radius = None
        self.log_message("Режим рисования круга активирован. ЛКМ - центр и край, ПКМ - завершить")

    def clear_polygon(self):
        """Очистка полигона - С ОТЛАДКОЙ"""

        self.polygon_points = []
        self.circle_center = None
        self.circle_radius = None
        self.video_label.polygon_points = []
        self.video_label.update()  # Перерисовываем виджет
        self.log_message("Круг очищен")

    def get_current_frame(self):
        """Получение текущего кадра (потокобезопасно)"""
        with QMutexLocker(self.mutex):
            return self.current_frame.copy() if self.current_frame is not None else None

    def compress_frame_to_640x480(self, frame):
        """Сжатие входного кадра до размера 640x480 с сохранением пропорций"""
        if frame is None:
            return None

        # Получаем размеры исходного кадра
        height, width = frame.shape[:2]

        # Целевые размеры
        target_width, target_height = 640, 480

        # Если кадр уже нужного размера, возвращаем как есть
        if width == target_width and height == target_height:
            return frame

        # Вычисляем коэффициент масштабирования
        scale_x = target_width / width
        scale_y = target_height / height
        scale = min(scale_x, scale_y)  # Используем минимальный для сохранения пропорций

        # Новые размеры с сохранением пропорций
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Масштабируем изображение
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Создаем черный фон 640x480
        output_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Вычисляем позицию для центрирования
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        # Помещаем масштабированное изображение в центр
        output_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

        return output_frame

    def resize_frame_to_640x480(self, frame):
        """Изменение размера кадра до 640x480 с сохранением пропорций (для записи)"""
        # Поскольку входное видео уже сжато до 640x480, просто возвращаем кадр
        return frame, 1.0, 0, 0

    def update_video_display(self):
        """Обновление отображения видео - С СЖАТЫМ ВХОДНЫМ ВИДЕО"""
        frame = self.get_current_frame()
        if frame is not None:
            display_frame = frame.copy()

            # Поскольку входное видео уже сжато до 640x480, работаем с этими размерами
            frame_height, frame_width = display_frame.shape[:2]

            # Получаем размеры отображаемого виджета
            label_size = self.video_label.size()
            widget_width = label_size.width()
            widget_height = label_size.height()

            # Вычисляем коэффициент масштабирования для отображения
            scale_x = frame_width / widget_width
            scale_y = frame_height / widget_height
            scale = max(scale_x, scale_y)

            # Детекция автомобилей если включена
            if self.vehicle_detection_enabled and self.yolo_detector:
                try:
                    if self.circle_center is not None and self.circle_radius is not None:
                        # Координаты круга уже в масштабе 640x480, используем их напрямую
                        circle_center = (
                            int(self.circle_center[0]),
                            int(self.circle_center[1])
                        )
                        circle_radius = int(self.circle_radius)

                        # Проверяем, что координаты круга в пределах кадра 640x480
                        circle_center = (
                            max(0, min(circle_center[0], frame_width - 1)),
                            max(0, min(circle_center[1], frame_height - 1))
                        )

                        # Детекция с кругом
                        annotated_frame, vehicles_in_circle = self.yolo_detector.detect_vehicles_in_circle(
                            display_frame, circle_center, circle_radius
                        )
                        vehicles = vehicles_in_circle
                    else:
                        # Детекция на всем кадре
                        annotated_frame, vehicles = self.yolo_detector.detect_vehicles(
                            display_frame, None, None
                        )

                    display_frame = annotated_frame

                    # Обновляем счетчик автомобилей
                    vehicle_count = len(vehicles)
                    if vehicle_count != self.vehicle_count:
                        self.vehicle_count = vehicle_count
                        self.vehicles_label.setText(str(self.vehicle_count))

                        if vehicle_count > 0:
                            # Подсчитываем по типам
                            counts = self.yolo_detector.count_vehicles_by_type(vehicles)
                            if self.circle_center is not None:
                                self.log_message(f"В круге обнаружено ТС: {counts['total']} "
                                                 f"(машины: {counts['car']}, грузовики: {counts['truck']}, "
                                                 f"автобусы: {counts['bus']}, мотоциклы: {counts['motorcycle']})")
                            else:
                                self.log_message(f"Обнаружено ТС: {counts['total']} "
                                                 f"(машины: {counts['car']}, грузовики: {counts['truck']}, "
                                                 f"автобусы: {counts['bus']}, мотоциклы: {counts['motorcycle']})")

                except Exception as e:
                    self.log_message(f"Ошибка YOLO детекции: {e}")

            # Запись видео если включена
            if self.is_recording and self.video_writer:
                try:
                    # Кадр уже в нужном разрешении 640x480, записываем напрямую
                    recording_frame, _, _, _ = self.resize_frame_to_640x480(display_frame)

                    # Записываем кадр
                    with QMutexLocker(self.recording_mutex):
                        if self.is_recording and self.video_writer:
                            self.video_writer.write(recording_frame)

                except Exception as e:
                    self.log_message(f"Ошибка записи кадра: {e}")

            # Конвертируем в QPixmap для отображения
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Масштабируем изображение с сохранением пропорций для виджета
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

        else:
            # Если видео не воспроизводится, показываем сообщение
            if hasattr(self, 'video_source') and self.video_source is not None:
                if not self.is_running:
                    self.video_label.setText("Видео остановлено")
                else:
                    self.video_label.setText("Загрузка видео...")
            else:
                self.video_label.setText("Выберите источник видео")

    def process_test_images(self):
        """Обработка изображений из каталога test_image с помощью YOLO"""
        if not self.yolo_available or not self.yolo_detector:
            self.log_message("YOLO детектор недоступен!")
            return

        try:
            self.log_message("Начинаем анализ изображений из каталога test_image...")
            results = self.yolo_detector.analyze_test_images(save_results=True)

            if results:
                total_images = len(results)
                total_vehicles = sum(r['total_vehicles'] for r in results)
                images_with_vehicles = sum(1 for r in results if r['total_vehicles'] > 0)

                self.log_message(f"Анализ завершен! Обработано {total_images} изображений")
                self.log_message(
                    f"Найдено {total_vehicles} транспортных средств на {images_with_vehicles} изображениях")
                self.log_message("Результаты сохранены в test_image/detection_results/")
            else:
                self.log_message("Не найдено изображений для анализа в каталоге test_image")

        except Exception as e:
            self.log_message(f"Ошибка при анализе изображений: {e}")

    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        # Останавливаем запись если она идет
        if self.is_recording:
            self.stop_recording()

        # Останавливаем анализ
        self.stop_analysis()

        event.accept()


def create_menu_bar(window):
    """Создание строки меню"""
    menubar = window.menuBar()

    # Меню "Файл"
    file_menu = menubar.addMenu('Файл')

    # Действие "Анализировать изображения"
    if window.yolo_available:
        analyze_action = file_menu.addAction('Анализировать изображения в test_image')
        analyze_action.triggered.connect(window.process_test_images)
        analyze_action.setToolTip('Запустить YOLO анализ всех изображений в каталоге test_image')

    # Действие "Выход"
    exit_action = file_menu.addAction('Выход')
    exit_action.triggered.connect(window.close)

    # Меню "Запись"
    record_menu = menubar.addMenu('Запись')

    start_record_action = record_menu.addAction('Начать запись видео')
    start_record_action.triggered.connect(window.start_recording)
    start_record_action.setToolTip('Начать запись видео детектирования в разрешении 640x480')

    stop_record_action = record_menu.addAction('Остановить запись видео')
    stop_record_action.triggered.connect(window.stop_recording)
    stop_record_action.setToolTip('Остановить запись видео')

    # Меню "Помощь"
    help_menu = menubar.addMenu('Помощь')

    about_action = help_menu.addAction('О программе')
    about_action.triggered.connect(lambda: window.log_message(
        "Детектор автомобилей с YOLO анализом\n"
        "Поддерживает YOLO детекцию автомобилей в реальном времени,\n"
        "пакетную обработку изображений и запись видео детектирования\n"
        "в разрешении 640x480 с рамками обнаружения и кругами зон"
    ))


def main():
    app = QApplication(sys.argv)

    # Настройка стиля приложения
    app.setStyle('Fusion')

    window = VideoAnalyzerGUI()

    # Добавляем строку меню
    create_menu_bar(window)

    window.show()

    # Выводим информацию о доступных функциях
    if window.yolo_available:
        window.log_message("YOLO детекция автомобилей доступна")
        window.log_message("Используйте меню 'Файл' -> 'Анализировать изображения' для пакетной обработки")
        window.log_message("Функция записи видео доступна (разрешение 640x480)")
    else:
        window.log_message("YOLO детекция недоступна - проверьте установку ultralytics")

    window.log_message("Выберите источник видео для детекции автомобилей")
    window.log_message("Входное видео автоматически сжимается до 640x480")
    window.log_message("Инструкция по записи:")
    window.log_message("1. Запустите анализ видео")
    window.log_message("2. При необходимости нарисуйте круг зоны детекции")
    window.log_message("3. Нажмите 'Записать видео' для начала записи")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()