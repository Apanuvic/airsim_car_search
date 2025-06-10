import sys, os, subprocess
import airsim
import numpy as np
import cv2
import csv
from datetime import datetime
from detector import CarDetector
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QGridLayout, QStatusBar
)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QShortcut

class DetectionThread(QThread):
    result_ready = pyqtSignal(np.ndarray, list)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.image = None
        self.running = True

    def run(self):
        while self.running:
            if self.image is not None:
                detections = self.detector.detect(self.image)
                self.result_ready.emit(self.image, detections)
                self.image = None

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def detect(self, image):
        self.image = image

class DroneApp(QWidget):
    def __init__(self):
        self.last_saved_position = None
        self.first_detection_time = None
        super().__init__()
        self.setWindowTitle("AirSim Drone Viewer")
        self.setGeometry(100, 100, 700, 580)
        self.setFocusPolicy(Qt.StrongFocus)

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-5, 3).join()

        self.detector = CarDetector()
        self.last_frame = None
        self.last_detections = []
        self.control_mode = "auto"
        self.yaw_angle = 0

        self.image_label = QLabel("Камера дрона")
        self.image_label.setFixedSize(640, 360)
        self.image_label.setScaledContents(True)

        self.mode_display = QLabel("Режим: Авто")
        self.mode_display.setAlignment(Qt.AlignCenter)

        self.status_bar = QStatusBar()
        self.coord_label = QLabel("Координаты: X=0, Y=0, Z=0")
        self.count_label = QLabel("Обнаружено машин: 0")
        self.status_label = QLabel("Статус: Подключено к AirSim")
        self.status_bar.addWidget(self.coord_label)
        self.status_bar.addWidget(self.count_label)
        self.status_bar.addWidget(self.status_label)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Авто", "Ручной"])
        self.mode_selector.currentTextChanged.connect(self.switch_mode)

        self.focus_button = QPushButton("Фокус")
        self.focus_button.clicked.connect(self.setFocus)

        self.start_button = QPushButton("Старт трансляции")
        self.start_button.clicked.connect(self.start_stream)

        self.save_button = QPushButton("Сделать снимок")
        self.save_button.clicked.connect(self.save_if_car_detected)

        self.report_button = QPushButton("Открыть отчёт")
        self.report_button.clicked.connect(self.open_report_file)

        self.patrol_button = QPushButton("Патруль")
        self.patrol_button.clicked.connect(self.run_patrol_mode)

        self.goto_button = QPushButton("Долететь до объекта")
        self.goto_button.clicked.connect(self.fly_to_detected_object)

        hbox = QHBoxLayout()
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.mode_selector)
        hbox.addWidget(self.focus_button)
        hbox.addWidget(self.save_button)
        hbox.addWidget(self.patrol_button)
        hbox.addWidget(self.goto_button)
        hbox.addWidget(self.report_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.mode_display)
        layout.addLayout(hbox)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(300)

        self.detection_thread = DetectionThread(self.detector)
        self.detection_thread.result_ready.connect(self.handle_detection_result)
        self.detection_thread.start()

        os.makedirs("images", exist_ok=True)
        self.log_path = "detection_log.csv"
        self.log_file = open(self.log_path, "w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["Timestamp", "X", "Y", "Z", "Confidence", "Object", "ImageName"])

        QShortcut(QKeySequence("W"), self, activated=lambda: self.manual_move(1, 0, 0))
        QShortcut(QKeySequence("S"), self, activated=lambda: self.manual_move(-1, 0, 0))
        QShortcut(QKeySequence("A"), self, activated=lambda: self.manual_move(0, -1, 0))
        QShortcut(QKeySequence("D"), self, activated=lambda: self.manual_move(0, 1, 0))
        QShortcut(QKeySequence("Up"), self, activated=lambda: self.manual_move(0, 0, -1))
        QShortcut(QKeySequence("Down"), self, activated=lambda: self.manual_move(0, 0, 1))
        QShortcut(QKeySequence("Q"), self, activated=lambda: self.rotate_drone(-15))
        QShortcut(QKeySequence("E"), self, activated=lambda: self.rotate_drone(15))

    def switch_mode(self, text):
        self.control_mode = "manual" if text == "Ручной" else "auto"
        self.mode_display.setText(f"Режим: {text}")
        self.setFocus()

    def manual_move(self, x, y, z):
        if self.control_mode != "manual":
            return
        v, t = 2, 0.5
        self.client.moveByVelocityAsync(v * x, v * y, v * z, t)

    def rotate_drone(self, delta_yaw):
        if self.control_mode != "manual":
            return
        self.yaw_angle = (self.yaw_angle + delta_yaw) % 360
        self.client.rotateToYawAsync(self.yaw_angle, 2)

    def start_stream(self):
        self.timer.start(300)

    def run_patrol_mode(self):
        points = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        for x, y in points:
            self.client.moveToPositionAsync(x, y, -5, 3).join()
            QTimer.singleShot(500, self.update_frame)

    def update_frame(self):
        try:
            if self.control_mode == "auto":
                self.yaw_angle = (self.yaw_angle + 10) % 360
                self.client.rotateToYawAsync(self.yaw_angle, 3)

            state = self.client.getMultirotorState()
            position = state.kinematics_estimated.position
            self.coord_label.setText(f"Координаты: X={position.x_val:.1f}, Y={position.y_val:.1f}, Z={position.z_val:.1f}")

            response = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])[0]
            if response.height == 0 or response.width == 0:
                self.status_label.setText("Статус: Пустое изображение")
                return
            self.status_label.setText("Статус: Получено изображение")

            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img = img1d.reshape(response.height, response.width, 3)
            self.last_frame = img.copy()
            self.detection_thread.detect(self.last_frame)

        except Exception as e:
            self.status_label.setText(f"Ошибка: {e}")

    def handle_detection_result(self, image, detections):
        self.last_detections = detections
        self.count_label.setText(f"Обнаружено машин: {len(detections)}")

        position = self.client.getMultirotorState().kinematics_estimated.position
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for (x1, y1, x2, y2, conf) in detections:
            if not self.first_detection_time:
                self.first_detection_time = datetime.now()

            distance_threshold = 3.0
            current_position = np.array([position.x_val, position.y_val, position.z_val])
            if self.last_saved_position is not None:
                dist = np.linalg.norm(current_position - self.last_saved_position)
                if dist < distance_threshold:
                    print("[SKIP] Повторное обнаружение слишком близко — не сохраняем.")
                    continue

            self.last_saved_position = current_position
            img_name = f"car_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            filepath = os.path.join("images", img_name)
            if image is not None:
                cv2.imwrite(filepath, image)
                print(f"[SAVE] Сохранено изображение: {filepath}")
            self.csv_writer.writerow([timestamp, position.x_val, position.y_val, position.z_val, conf, "Car", img_name])
            self.log_file.flush()
            print(f"[LOG] Запись в отчет: Car @ ({position.x_val:.2f}, {position.y_val:.2f}, {position.z_val:.2f}) conf={conf:.2f}")

        if image is None or image.size == 0:
            print("[WARNING] Пустое изображение для отрисовки")
            return

        img_draw = image.copy()
        for (x1, y1, x2, y2, conf) in detections:
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_draw, f"Car {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qt_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def save_if_car_detected(self):
        if self.last_frame is not None and self.last_detections:
            filename = os.path.join("images", f"car_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
            cv2.imwrite(filename, self.last_frame)

    def fly_to_detected_object(self):
        if not self.last_detections:
            self.status_label.setText("Статус: Нет цели")
            return
        x1, y1, x2, y2, conf = self.last_detections[0]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        current_pos = self.client.getMultirotorState().kinematics_estimated.position
        self.client.moveToPositionAsync(current_pos.x_val + 5, current_pos.y_val, -5, 2)

    def open_report_file(self):
        try:
            if sys.platform == "win32":
                os.startfile(self.log_path)
            elif sys.platform == "darwin":
                subprocess.call(["open", self.log_path])
            else:
                subprocess.call(["xdg-open", self.log_path])
        except Exception as e:
            print(f"[ERROR] Не удалось открыть отчет: {e}")

    def closeEvent(self, event):
        self.detection_thread.stop()
        self.log_file.close()
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DroneApp()
    viewer.show()
    sys.exit(app.exec_())
