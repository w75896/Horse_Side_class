import sys
import cv2
import numpy as np
import threading
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# 引入你提供的 OptimizedFaceTracker 與 apply_smart_mosaic
# 假設你將它們放在 face_tracker_module.py 中，這裡以匯入方式使用
from face_tracker_module import OptimizedFaceTracker, apply_smart_mosaic

class VideoMosaicApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("即時人臉馬賽克 GUI")
        self.resize(800, 600)

        self.label = QLabel("按下『開始』以啟動攝影機")
        self.label.setAlignment(Qt.AlignCenter)

        self.btn_start = QPushButton("開始")
        self.btn_start.clicked.connect(self.toggle_camera)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(5)
        self.slider.setMaximum(50)
        self.slider.setValue(15)
        self.slider.valueChanged.connect(self.update_mosaic_size)

        self.style_box = QComboBox()
        self.style_box.addItems(['pixelate', 'blur', 'black'])
        self.style_box.currentTextChanged.connect(self.update_mosaic_style)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(QLabel("強度:"))
        control_layout.addWidget(self.slider)
        control_layout.addWidget(QLabel("樣式:"))
        control_layout.addWidget(self.style_box)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(control_layout)
        self.setLayout(layout)

        # 初始化追蹤器與控制參數
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.tracker = OptimizedFaceTracker()
        self.mosaic_size = 15
        self.mosaic_style = 'pixelate'

        self.running = False

    def toggle_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.label.setText("無法開啟攝影機")
                return
            self.running = True
            self.btn_start.setText("停止")
            self.timer.start(30)  # 約每 30ms 更新一幀
        else:
            self.running = False
            self.btn_start.setText("開始")
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.label.setPixmap(QPixmap())  # 清空畫面

    def update_mosaic_size(self, value):
        self.mosaic_size = value

    def update_mosaic_style(self, style):
        self.mosaic_style = style

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        faces = self.tracker.update_face_tracking(frame)
        frame = apply_smart_mosaic(frame, faces, self.mosaic_size, self.mosaic_style)

        # 顯示處理資訊
        cv2.putText(frame, f'Mosaic: {self.mosaic_style} ({self.mosaic_size})',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f'Faces: {len(faces)}', 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoMosaicApp()
    win.show()
    sys.exit(app.exec_())
