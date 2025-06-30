import time
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                             QLabel, QSpinBox, QTextEdit, QGroupBox, QGridLayout,
                             QSlider, QFileDialog, QCheckBox)
from PyQt5.QtCore import Qt


def init_ui(window):
    """
    Инициализация пользовательского интерфейса

    Args:
        window: Экземпляр главного окна VideoAnalyzerGUI
    """
    window.setWindowTitle("Детектор автомобилей с YOLO + PyQt5")
    window.setGeometry(100, 100, 1200, 800)

    central_widget = QWidget()
    window.setCentralWidget(central_widget)

    main_layout = QHBoxLayout()
    central_widget.setLayout(main_layout)

    # Левая панель - видео
    video_layout = _create_video_panel(window)

    # Правая панель - настройки и результаты
    right_panel = _create_control_panel(window)

    # Добавляем панели в основной layout
    main_layout.addLayout(video_layout, 2)
    main_layout.addLayout(right_panel, 1)

    window.log_message("Приложение запущено")


def _create_video_panel(window):
    """
    Создание видео панели

    Args:
        window: Экземпляр главного окна

    Returns:
        QVBoxLayout: Layout видео панели
    """
    video_layout = QVBoxLayout()

    # Область отображения видео - занимает максимум места
    window.video_label = QLabel("Выберите источник видео")
    window.video_label.setMinimumSize(640, 480)
    window.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
    window.video_label.setAlignment(Qt.AlignCenter)
    window.video_label.setScaledContents(False)  # Важно для корректного масштабирования

    # Добавляем видео с растяжением
    video_layout.addWidget(window.video_label, 1)  # stretch factor = 1

    # Кнопки управления видео внизу
    video_controls = _create_video_controls(window)
    video_layout.addLayout(video_controls, 0)  # stretch factor = 0

    return video_layout


def _create_video_controls(window):
    """
    Создание элементов управления видео

    Args:
        window: Экземпляр главного окна

    Returns:
        QHBoxLayout: Layout с кнопками управления
    """
    video_controls = QHBoxLayout()

    # Создание кнопок
    window.select_file_btn = QPushButton("Выбрать файл")
    window.select_camera_btn = QPushButton("Камера")
    window.start_btn = QPushButton("Запустить")
    window.stop_btn = QPushButton("Остановить")
    window.record_btn = QPushButton("Записать видео")  # Новая кнопка
    window.stop_record_btn = QPushButton("Остановить запись")  # Новая кнопка

    # Подключение сигналов
    window.select_file_btn.clicked.connect(window.select_video_file)
    window.select_camera_btn.clicked.connect(window.select_camera)
    window.start_btn.clicked.connect(window.start_analysis)
    window.stop_btn.clicked.connect(window.stop_analysis)
    window.record_btn.clicked.connect(window.start_recording)  # Новый сигнал
    window.stop_record_btn.clicked.connect(window.stop_recording)  # Новый сигнал

    # Настройка видимости кнопок записи
    window.stop_record_btn.setEnabled(False)

    # Добавление кнопок в layout
    video_controls.addWidget(window.select_file_btn)
    video_controls.addWidget(window.select_camera_btn)
    video_controls.addWidget(window.start_btn)
    video_controls.addWidget(window.stop_btn)
    video_controls.addWidget(window.record_btn)
    video_controls.addWidget(window.stop_record_btn)

    return video_controls


def _create_control_panel(window):
    """
    Создание правой панели управления

    Args:
        window: Экземпляр главного окна

    Returns:
        QVBoxLayout: Layout правой панели
    """
    right_panel = QVBoxLayout()

    # Группа настроек
    settings_group = _create_settings_group(window)
    right_panel.addWidget(settings_group)

    # Группа статистики
    stats_group = _create_stats_group(window)
    right_panel.addWidget(stats_group)

    # Группа лога событий
    log_group = _create_log_group(window)
    right_panel.addWidget(log_group)

    return right_panel


def _create_settings_group(window):
    """
    Создание группы настроек с настройками YOLO детекции

    Args:
        window: Экземпляр главного окна

    Returns:
        QGroupBox: Группа с настройками
    """
    settings_group = QGroupBox("Настройки")
    settings_layout = QGridLayout()

    # YOLO детекция
    window.vehicle_detection_checkbox = QCheckBox("Детекция автомобилей (YOLO)")
    window.vehicle_detection_checkbox.setChecked(True)
    window.vehicle_detection_checkbox.stateChanged.connect(window.toggle_vehicle_detection)
    settings_layout.addWidget(window.vehicle_detection_checkbox, 0, 0, 1, 3)

    # Порог уверенности YOLO
    settings_layout.addWidget(QLabel("Порог уверенности YOLO:"), 1, 0)
    window.yolo_confidence_slider = QSlider(Qt.Horizontal)
    window.yolo_confidence_slider.setRange(10, 90)
    window.yolo_confidence_slider.setValue(50)
    window.yolo_confidence_slider.valueChanged.connect(window.update_yolo_confidence)
    window.yolo_confidence_label = QLabel("0.50")
    settings_layout.addWidget(window.yolo_confidence_slider, 1, 1)
    settings_layout.addWidget(window.yolo_confidence_label, 1, 2)

    # ID камеры
    settings_layout.addWidget(QLabel("ID камеры:"), 2, 0)
    window.camera_id_spin = QSpinBox()
    window.camera_id_spin.setRange(0, 10)
    settings_layout.addWidget(window.camera_id_spin, 2, 1)

    # Кнопки для создания круга
    window.polygon_btn = QPushButton("Нарисовать круг")
    window.polygon_btn.clicked.connect(window.start_polygon_drawing)
    settings_layout.addWidget(window.polygon_btn, 3, 0, 1, 2)

    window.clear_polygon_btn = QPushButton("Очистить круг")
    window.clear_polygon_btn.clicked.connect(window.clear_polygon)
    settings_layout.addWidget(window.clear_polygon_btn, 4, 0, 1, 2)

    settings_group.setLayout(settings_layout)
    return settings_group


def _create_stats_group(window):
    """
    Создание группы статистики

    Args:
        window: Экземпляр главного окна

    Returns:
        QGroupBox: Группа со статистикой
    """
    stats_group = QGroupBox("Статистика")
    stats_layout = QGridLayout()

    # Счетчик автомобилей
    stats_layout.addWidget(QLabel("Автомобилей обнаружено:"), 0, 0)
    window.vehicles_label = QLabel("0")
    stats_layout.addWidget(window.vehicles_label, 0, 1)

    # Статус
    stats_layout.addWidget(QLabel("Статус:"), 1, 0)
    window.status_label = QLabel("Остановлен")
    stats_layout.addWidget(window.status_label, 1, 1)

    # Статус записи
    stats_layout.addWidget(QLabel("Запись:"), 2, 0)
    window.record_status_label = QLabel("Не записывается")
    stats_layout.addWidget(window.record_status_label, 2, 1)

    stats_group.setLayout(stats_layout)
    return stats_group


def _create_log_group(window):
    """
    Создание группы лога событий

    Args:
        window: Экземпляр главного окна

    Returns:
        QGroupBox: Группа с логом
    """
    log_group = QGroupBox("Лог событий")
    log_layout = QVBoxLayout()

    window.log_text = QTextEdit()
    window.log_text.setMaximumHeight(200)
    log_layout.addWidget(window.log_text)

    log_group.setLayout(log_layout)
    return log_group