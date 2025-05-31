import sys
import cv2
import numpy as np
import threading
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QComboBox, QMessageBox, QGroupBox,
    QCheckBox, QSpinBox, QStatusBar, QMainWindow, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

# 引入你提供的 OptimizedFaceTracker 與 apply_smart_mosaic
from face_tracker_module import OptimizedFaceTracker, apply_smart_mosaic

class VideoThread(QThread):
    """獨立的視頻處理線程，避免界面卡頓"""
    frame_ready = pyqtSignal(np.ndarray)
    stats_ready = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self.tracker = OptimizedFaceTracker()
        self.mosaic_size = 15
        self.mosaic_style = 'pixelate'
        self.paused = False
        self.show_debug = False
        
        # 性能監控
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
    def set_camera(self, camera_index):
        """設置攝像頭"""
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(camera_index)
        if self.cap.isOpened():
            # 優化攝像頭設置
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return True
        return False
    
    def update_settings(self, mosaic_size, mosaic_style):
        """更新馬賽克設置"""
        self.mosaic_size = mosaic_size
        self.mosaic_style = mosaic_style
    
    def toggle_pause(self):
        """切換暫停狀態"""
        self.paused = not self.paused
    
    def toggle_debug(self, show_debug):
        """切換調試顯示"""
        self.show_debug = show_debug
    
    def run(self):
        """主線程循環"""
        self.running = True
        
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            if self.paused:
                time.sleep(0.1)
                continue
            
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # 水平翻轉（鏡像效果）
            # frame = cv2.flip(frame, 1)
            
            # 人臉檢測和追蹤
            faces = self.tracker.update_face_tracking(frame)
            
            # 應用馬賽克
            processed_frame = apply_smart_mosaic(frame.copy(), faces, 
                                               self.mosaic_size, self.mosaic_style)
            
            # 添加調試信息
            if self.show_debug:
                self.add_debug_info(processed_frame, faces)
            
            # 計算 FPS
            frame_time = time.time() - frame_start
            self.fps_counter += 1
            if self.fps_counter >= 15:
                self.current_fps = 15 / (time.time() - self.fps_timer)
                self.fps_timer = time.time()
                self.fps_counter = 0
                
                # 發送統計信息
                stats = {
                    'fps': self.current_fps,
                    'faces': len(faces),
                    'trackers': len(self.tracker.trackers),
                    'processing_time': frame_time * 1000,
                    'detection_method': self.tracker.detection_method
                }
                self.stats_ready.emit(stats)
            
            # 發送處理後的幀
            self.frame_ready.emit(processed_frame)
    
    def add_debug_info(self, frame, faces):
        """添加調試信息到幀上"""
        # 繪製人臉框
        for i, face in enumerate(faces):
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Face {i+1}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 顯示基本信息
        info_y = 25
        cv2.putText(frame, f'Method: {self.tracker.detection_method}',
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        info_y += 25
        cv2.putText(frame, f'Faces: {len(faces)} | Trackers: {len(self.tracker.trackers)}',
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    def stop(self):
        """停止線程"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()

class VideoMosaicApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_video_thread()
        
    def init_ui(self):
        """初始化用戶界面"""
        self.setWindowTitle("高性能即時人臉馬賽克 - YOLOv11n 增強版")
        self.setGeometry(100, 100, 900, 700)
        
        # 設置樣式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #606060;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QSlider::groove:horizontal {
                border: 1px solid #606060;
                height: 8px;
                background: #404040;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QComboBox {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 5px;
            }
            QGroupBox {
                color: #ffffff;
                border: 2px solid #606060;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # 創建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局
        main_layout = QHBoxLayout(central_widget)
        
        # 左側視頻顯示區域
        video_layout = QVBoxLayout()
        
        # 視頻顯示標籤
        self.video_label = QLabel("按下『開始』以啟動攝像頭")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #606060;
                background-color: #1a1a1a;
                font-size: 16px;
            }
        """)
        
        # 控制按鈕
        button_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("🎥 開始")
        self.btn_start.clicked.connect(self.toggle_camera)
        
        self.btn_pause = QPushButton("⏸️ 暫停")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.setEnabled(False)
        
        self.btn_screenshot = QPushButton("📷 截圖")
        self.btn_screenshot.clicked.connect(self.take_screenshot)
        self.btn_screenshot.setEnabled(False)
        
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_pause)
        button_layout.addWidget(self.btn_screenshot)
        button_layout.addStretch()
        
        video_layout.addWidget(self.video_label)
        video_layout.addLayout(button_layout)
        
        # 右側控制面板
        control_panel = self.create_control_panel()
        
        # 添加到主佈局
        main_layout.addLayout(video_layout, 2)
        main_layout.addWidget(control_panel, 1)
        
        # 狀態欄
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就緒")
        
        # 存儲變量
        self.running = False
        self.current_frame = None
        
    def create_control_panel(self):
        """創建控制面板"""
        panel = QFrame()
        panel.setMaximumWidth(300)
        panel.setStyleSheet("QFrame { border: 1px solid #606060; }")
        
        layout = QVBoxLayout(panel)
        
        # 馬賽克設置組
        mosaic_group = QGroupBox("馬賽克設置")
        mosaic_layout = QVBoxLayout(mosaic_group)
        
        # 強度滑塊
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("強度:"))
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(3, 50)
        self.intensity_slider.setValue(15)
        self.intensity_slider.valueChanged.connect(self.update_mosaic_settings)
        self.intensity_label = QLabel("15")
        intensity_layout.addWidget(self.intensity_slider)
        intensity_layout.addWidget(self.intensity_label)
        
        # 樣式選擇
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("樣式:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(['pixelate', 'blur', 'black'])
        self.style_combo.currentTextChanged.connect(self.update_mosaic_settings)
        style_layout.addWidget(self.style_combo)
        
        mosaic_layout.addLayout(intensity_layout)
        mosaic_layout.addLayout(style_layout)
        
        # 攝像頭設置組
        camera_group = QGroupBox("攝像頭設置")
        camera_layout = QVBoxLayout(camera_group)
        
        # 攝像頭選擇
        camera_layout.addWidget(QLabel("攝像頭索引:"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 5)
        self.camera_spin.setValue(0)
        camera_layout.addWidget(self.camera_spin)
        
        # 調試選項
        debug_group = QGroupBox("調試選項")
        debug_layout = QVBoxLayout(debug_group)
        
        self.debug_check = QCheckBox("顯示人臉框")
        self.debug_check.toggled.connect(self.toggle_debug)
        debug_layout.addWidget(self.debug_check)
        
        # 檢測方法切換
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("檢測方法:"))
        self.method_combo = QComboBox()
        method_layout.addWidget(self.method_combo)
        debug_layout.addLayout(method_layout)
        
        # 統計信息組
        stats_group = QGroupBox("實時統計")
        stats_layout = QVBoxLayout(stats_group)
        
        self.fps_label = QLabel("FPS: --")
        self.faces_label = QLabel("檢測到的人臉: --")
        self.trackers_label = QLabel("活躍追蹤器: --")
        self.processing_label = QLabel("處理時間: --")
        self.method_label = QLabel("檢測方法: --")
        
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.faces_label)
        stats_layout.addWidget(self.trackers_label)
        stats_layout.addWidget(self.processing_label)
        stats_layout.addWidget(self.method_label)
        
        # 添加所有組到面板
        layout.addWidget(mosaic_group)
        layout.addWidget(camera_group)
        layout.addWidget(debug_group)
        layout.addWidget(stats_group)
        layout.addStretch()
        
        return panel
    
    def init_video_thread(self):
        """初始化視頻處理線程"""
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_video_display)
        self.video_thread.stats_ready.connect(self.update_stats)
        
        # 初始化檢測方法選項
        self.update_method_combo()
    
    def update_method_combo(self):
        """更新檢測方法下拉框"""
        methods = []
        if hasattr(self.video_thread.tracker, 'yolo_available') and self.video_thread.tracker.yolo_available:
            methods.append('YOLO')
        
        # 檢查 MediaPipe 是否可用
        try:
            import mediapipe as mp
            methods.append('MediaPipe')
        except ImportError:
            pass
        
        methods.append('Haar Cascade')
        
        self.method_combo.clear()
        self.method_combo.addItems(methods)
        self.method_combo.currentTextChanged.connect(self.change_detection_method)
    
    def toggle_camera(self):
        """切換攝像頭狀態"""
        if not self.running:
            camera_index = self.camera_spin.value()
            if self.video_thread.set_camera(camera_index):
                self.video_thread.start()
                self.running = True
                self.btn_start.setText("🛑 停止")
                self.btn_pause.setEnabled(True)
                self.btn_screenshot.setEnabled(True)
                self.status_bar.showMessage("攝像頭已啟動")
            else:
                QMessageBox.warning(self, "錯誤", f"無法開啟攝像頭 {camera_index}")
        else:
            self.video_thread.stop()
            self.running = False
            self.btn_start.setText("🎥 開始")
            self.btn_pause.setEnabled(False)
            self.btn_screenshot.setEnabled(False)
            self.video_label.setText("按下『開始』以啟動攝像頭")
            self.status_bar.showMessage("攝像頭已停止")
    
    def toggle_pause(self):
        """切換暫停狀態"""
        if self.running:
            self.video_thread.toggle_pause()
            if self.video_thread.paused:
                self.btn_pause.setText("▶️ 繼續")
                self.status_bar.showMessage("已暫停")
            else:
                self.btn_pause.setText("⏸️ 暫停")
                self.status_bar.showMessage("攝像頭運行中")
    
    def take_screenshot(self):
        """截圖功能"""
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, self.current_frame)
            QMessageBox.information(self, "截圖成功", f"已保存為 {filename}")
    
    def update_mosaic_settings(self):
        """更新馬賽克設置"""
        intensity = self.intensity_slider.value()
        style = self.style_combo.currentText()
        
        self.intensity_label.setText(str(intensity))
        
        if self.running:
            self.video_thread.update_settings(intensity, style)
    
    def toggle_debug(self, enabled):
        """切換調試顯示"""
        if self.running:
            self.video_thread.toggle_debug(enabled)
    
    def change_detection_method(self, method_name):
        """更改檢測方法"""
        method_map = {
            'YOLO': 'yolo',
            'MediaPipe': 'mediapipe',
            'Haar Cascade': 'haar'
        }
        
        if method_name in method_map and self.running:
            self.video_thread.tracker.detection_method = method_map[method_name]
    
    def update_video_display(self, frame):
        """更新視頻顯示"""
        self.current_frame = frame
        
        # 轉換為 Qt 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 縮放到合適大小
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_stats(self, stats):
        """更新統計信息"""
        self.fps_label.setText(f"FPS: {stats['fps']:.1f}")
        self.faces_label.setText(f"檢測到的人臉: {stats['faces']}")
        self.trackers_label.setText(f"活躍追蹤器: {stats['trackers']}")
        self.processing_label.setText(f"處理時間: {stats['processing_time']:.1f}ms")
        self.method_label.setText(f"檢測方法: {stats['detection_method'].upper()}")
    
    def closeEvent(self, event):
        """關閉事件處理"""
        if self.running:
            self.video_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # 設置應用程式圖標和基本信息
    app.setApplicationName("人臉馬賽克")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("AI Vision Tools")
    
    # 創建並顯示主窗口
    window = VideoMosaicApp()
    window.show()
    
    # 顯示歡迎信息
    QMessageBox.information(window, "歡迎使用", 
                           "高性能即時人臉馬賽克應用程式\n\n"
                           "功能特色:\n"
                           "• YOLOv11n 深度學習檢測\n"
                           "• 智能追蹤算法\n"
                           "• 多種馬賽克效果\n"
                           "• 實時性能監控\n\n"
                           "請確保已安裝相關依賴套件")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()