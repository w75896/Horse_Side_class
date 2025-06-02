import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import os
from datetime import datetime

# 導入主程式模組
try:
    from face_tracker_module import (
        OptimizedFaceTracker, 
        VideoRecorder, 
        apply_smart_mosaic,
        DEEPFACE_AVAILABLE,
        MEDIAPIPE_AVAILABLE
    )
except ImportError:
    print("錯誤：無法導入 paste.py 模組")
    print("請確保 paste.py 在同一目錄下")
    sys.exit(1)

class VideoThread(QThread):
    """影像處理執行緒"""
    changePixmap = pyqtSignal(QImage)
    updateFPS = pyqtSignal(float)
    updateFaceCount = pyqtSignal(int)
    updateRecordingInfo = pyqtSignal(dict)
    errorOccurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.camera_index = 0  # 預設使用第一個相機
        self.tracker = OptimizedFaceTracker()
        self.recorder = VideoRecorder()
        self.running = True
        self.paused = False
        
        # 參數
        self.mosaic_size = 15
        self.mosaic_style = 'pixelate'
        self.faces = []
        
        # FPS 計算
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
    def run(self):
        """執行緒主迴圈"""
        # 嘗試開啟相機
        if isinstance(self.camera_index, str):
            # 如果是檔案路徑
            self.cap = cv2.VideoCapture(self.camera_index)
        else:
            # 如果是相機索引
            self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            self.errorOccurred.emit(f"無法開啟相機或影片: {self.camera_index}")
            return
        
        # 優化攝影機設定（只對實體相機設定）
        if isinstance(self.camera_index, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while self.running:
            if not self.paused:
                ret, frame = self.cap.read()
                if ret:
                    # 水平翻轉（只對實體相機）
                    if isinstance(self.camera_index, int):
                        frame = cv2.flip(frame, 1)
                    
                    # 人臉檢測
                    self.faces = self.tracker.update_face_detection(frame)
                    
                    # 更新人臉數量
                    self.updateFaceCount.emit(len(self.faces))
                    
                    # 應用馬賽克
                    display_frame = apply_smart_mosaic(
                        frame.copy(), 
                        self.faces, 
                        self.mosaic_size, 
                        self.mosaic_style,
                        self.tracker
                    )
                    
                    # 錄影
                    if self.recorder.recording:
                        self.recorder.update_fps(self.current_fps)
                        self.recorder.write_frame(display_frame)
                        
                        # 更新錄影資訊
                        rec_info = self.recorder.get_recording_info()
                        if rec_info:
                            self.updateRecordingInfo.emit(rec_info)
                    
                    # 計算 FPS
                    self.fps_counter += 1
                    if self.fps_counter >= 15:
                        self.current_fps = 15 / (time.time() - self.fps_timer)
                        self.fps_timer = time.time()
                        self.fps_counter = 0
                        self.updateFPS.emit(self.current_fps)
                    
                    # 轉換為 Qt 格式
                    rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    self.changePixmap.emit(qt_image)
                else:
                    # 影片播放結束
                    if isinstance(self.camera_index, str):
                        break
            else:
                self.msleep(50)
    
    def stop(self):
        """停止執行緒"""
        self.running = False
        if self.recorder.recording:
            self.recorder.stop_recording()
        if self.cap:
            self.cap.release()
        self.wait()

class FaceMosaicGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.video_thread = None
        self.available_cameras = []  # 儲存可用相機列表
        
    def refresh_cameras(self):
        """重新整理可用相機列表"""
        self.camera_combo.clear()
        self.available_cameras = []
        
        # 測試可用的相機
        for i in range(10):  # 測試前10個索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # 取得相機資訊
                backend = cap.getBackendName()
                self.available_cameras.append(i)
                self.camera_combo.addItem(f"相機 {i} ({backend})")
                cap.release()
        
        # 如果沒有找到相機
        if not self.available_cameras:
            self.camera_combo.addItem("未偵測到相機")
        
        # 添加分隔線
        if self.available_cameras:
            self.camera_combo.insertSeparator(self.camera_combo.count())
        
        # 添加檔案選項
        self.camera_combo.addItem("選擇影片檔案...")
    
    def select_video_file(self):
        """選擇影片檔案"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇影片檔案",
            "",
            "影片檔案 (*.mp4 *.avi *.mov *.mkv *.flv);;所有檔案 (*.*)"
        )
        
        if file_path:
            # 添加到下拉選單
            display_name = f"檔案: {os.path.basename(file_path)}"
            
            # 檢查是否已存在
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemText(i) == display_name:
                    self.camera_combo.setCurrentIndex(i)
                    return
            
            # 添加新項目
            self.camera_combo.addItem(display_name)
            self.camera_combo.setItemData(self.camera_combo.count() - 1, file_path)
            self.camera_combo.setCurrentIndex(self.camera_combo.count() - 1)
    
    def get_selected_camera(self):
        """取得選擇的相機或檔案"""
        current_index = self.camera_combo.currentIndex()
        current_text = self.camera_combo.currentText()
        
        if current_text == "未偵測到相機":
            return None
        elif current_text == "選擇影片檔案...":
            self.select_video_file()
            return None
        elif current_text.startswith("檔案:"):
            # 返回檔案路徑
            return self.camera_combo.itemData(current_index)
        else:
            # 返回相機索引
            if current_index < len(self.available_cameras):
                return self.available_cameras[current_index]
        
        return None
        
    def initUI(self):
        """初始化使用者介面"""
        self.setWindowTitle('智能人臉馬賽克系統')
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        
        # 設定圖標
        self.setWindowIcon(QIcon())
        
        # 中央元件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局
        main_layout = QHBoxLayout(central_widget)
        
        # 左側：影像顯示區
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 影像標籤
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 10px;
                background-color: #000;
            }
        """)
        self.image_label.setScaledContents(True)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # 在未啟動時顯示提示
        self.image_label.setText("點擊「啟動相機」開始")
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 10px;
                background-color: #000;
                color: #fff;
                font-size: 20px;
            }
        """)
        
        left_layout.addWidget(self.image_label)
        
        # 狀態列
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        
        self.fps_label = QLabel('FPS: 0.0')
        self.face_count_label = QLabel('檢測到的臉部: 0')
        self.recording_label = QLabel('未錄影')
        self.recording_label.setStyleSheet("color: green;")
        
        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.face_count_label)
        status_layout.addWidget(self.recording_label)
        status_layout.addStretch()
        
        left_layout.addWidget(status_widget)
        
        # 右側：控制面板
        right_panel = QWidget()
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        
        # 標題
        title_label = QLabel('控制面板')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
        """)
        right_layout.addWidget(title_label)
        
        # 滾動區域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 1. 相機控制
        camera_group = self.create_camera_controls()
        scroll_layout.addWidget(camera_group)
        
        # 2. 馬賽克設定
        mosaic_group = self.create_mosaic_controls()
        scroll_layout.addWidget(mosaic_group)
        
        # 3. 檢測設定
        detection_group = self.create_detection_controls()
        scroll_layout.addWidget(detection_group)
        
        # 4. 小孩保護
        if DEEPFACE_AVAILABLE:
            child_group = self.create_child_protection_controls()
            scroll_layout.addWidget(child_group)
        
        # 5. 錄影控制
        recording_group = self.create_recording_controls()
        scroll_layout.addWidget(recording_group)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        
        right_layout.addWidget(scroll_area)
        
        # 添加到主佈局
        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(right_panel, 1)
        
        # 設定樣式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                padding: 8px;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #d3d3d3;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        
    def create_camera_controls(self):
        """建立相機控制元件"""
        group = QGroupBox("相機控制")
        layout = QVBoxLayout()
        
        # 相機選擇
        camera_label = QLabel("選擇相機/影片:")
        layout.addWidget(camera_label)
        
        camera_select_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.refresh_cameras()
        camera_select_layout.addWidget(self.camera_combo)
        
        # 重新整理按鈕
        refresh_button = QPushButton("🔄")
        refresh_button.setMaximumWidth(30)
        refresh_button.setToolTip("重新整理相機列表")
        refresh_button.clicked.connect(self.refresh_cameras)
        camera_select_layout.addWidget(refresh_button)
        
        # 選擇檔案按鈕
        file_button = QPushButton("📁")
        file_button.setMaximumWidth(30)
        file_button.setToolTip("選擇影片檔案")
        file_button.clicked.connect(self.select_video_file)
        camera_select_layout.addWidget(file_button)
        
        layout.addLayout(camera_select_layout)
        
        # 啟動/停止按鈕
        self.start_button = QPushButton('啟動相機')
        self.start_button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.start_button)
        
        # 暫停按鈕
        self.pause_button = QPushButton('暫停')
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        layout.addWidget(self.pause_button)
        
        group.setLayout(layout)
        return group
    
    def create_mosaic_controls(self):
        """建立馬賽克控制元件"""
        group = QGroupBox("馬賽克設定")
        layout = QVBoxLayout()
        
        # 效果選擇
        effect_label = QLabel("效果類型:")
        layout.addWidget(effect_label)
        
        self.effect_combo = QComboBox()
        self.effect_combo.addItems(['像素化', '模糊', '黑色遮擋'])
        self.effect_combo.currentTextChanged.connect(self.change_mosaic_style)
        layout.addWidget(self.effect_combo)
        
        # 強度調整
        intensity_label = QLabel("馬賽克強度:")
        layout.addWidget(intensity_label)
        
        intensity_layout = QHBoxLayout()
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(3)
        self.intensity_slider.setMaximum(50)
        self.intensity_slider.setValue(15)
        self.intensity_slider.valueChanged.connect(self.change_mosaic_size)
        
        self.intensity_value = QLabel('15')
        
        intensity_layout.addWidget(self.intensity_slider)
        intensity_layout.addWidget(self.intensity_value)
        layout.addLayout(intensity_layout)
        
        # 安全邊界
        margin_label = QLabel("安全邊界:")
        layout.addWidget(margin_label)
        
        margin_layout = QHBoxLayout()
        self.margin_slider = QSlider(Qt.Horizontal)
        self.margin_slider.setMinimum(100)
        self.margin_slider.setMaximum(200)
        self.margin_slider.setValue(130)
        self.margin_slider.valueChanged.connect(self.change_safety_margin)
        
        self.margin_value = QLabel('1.3x')
        
        margin_layout.addWidget(self.margin_slider)
        margin_layout.addWidget(self.margin_value)
        layout.addLayout(margin_layout)
        
        group.setLayout(layout)
        return group
    
    def create_detection_controls(self):
        """建立檢測控制元件"""
        group = QGroupBox("檢測設定")
        layout = QVBoxLayout()
        
        # 檢測方法
        method_label = QLabel("檢測方法:")
        layout.addWidget(method_label)
        
        self.method_combo = QComboBox()
        
        # 添加可用的檢測方法
        methods = []
        if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.tracker.yolo_available:
            methods.append('YOLO')
        if MEDIAPIPE_AVAILABLE:
            methods.append('MediaPipe')
        methods.append('Haar Cascade')
        
        self.method_combo.addItems(methods)
        self.method_combo.currentTextChanged.connect(self.change_detection_method)
        layout.addWidget(self.method_combo)
        
        # 檢測資訊
        info_label = QLabel("可用檢測方法將在啟動相機後顯示")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group
    
    def create_child_protection_controls(self):
        """建立小孩保護控制元件"""
        group = QGroupBox("小孩保護功能")
        layout = QVBoxLayout()
        
        # 啟用開關
        self.child_protection_check = QCheckBox("啟用小孩保護")
        self.child_protection_check.stateChanged.connect(self.toggle_child_protection)
        layout.addWidget(self.child_protection_check)
        
        # 年齡閾值
        age_label = QLabel("年齡閾值:")
        layout.addWidget(age_label)
        
        age_layout = QHBoxLayout()
        self.age_spinbox = QSpinBox()
        self.age_spinbox.setMinimum(1)
        self.age_spinbox.setMaximum(100)
        self.age_spinbox.setValue(18)
        self.age_spinbox.valueChanged.connect(self.change_age_threshold)
        
        age_layout.addWidget(self.age_spinbox)
        age_layout.addWidget(QLabel("歲"))
        age_layout.addStretch()
        layout.addLayout(age_layout)
        
        # 說明
        info_label = QLabel("啟用後只會對設定年齡以下的臉部應用馬賽克")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group
    
    def create_recording_controls(self):
        """建立錄影控制元件"""
        group = QGroupBox("錄影功能")
        layout = QVBoxLayout()
        
        # 錄影按鈕
        self.record_button = QPushButton('開始錄影')
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        layout.addWidget(self.record_button)
        
        # 錄影資訊
        self.recording_info_label = QLabel("錄影資訊將在此顯示")
        self.recording_info_label.setWordWrap(True)
        self.recording_info_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.recording_info_label)
        
        # 開啟資料夾按鈕
        self.open_folder_button = QPushButton('開啟輸出資料夾')
        self.open_folder_button.clicked.connect(self.open_output_folder)
        layout.addWidget(self.open_folder_button)
        
        group.setLayout(layout)
        return group
    
    def toggle_camera(self):
        """啟動/停止相機"""
        if self.video_thread is None:
            # 取得選擇的相機
            camera_source = self.get_selected_camera()
            if camera_source is None:
                return
            
            # 啟動相機
            self.video_thread = VideoThread()
            self.video_thread.camera_index = camera_source  # 設定相機來源
            self.video_thread.changePixmap.connect(self.update_image)
            self.video_thread.updateFPS.connect(self.update_fps)
            self.video_thread.updateFaceCount.connect(self.update_face_count)
            self.video_thread.updateRecordingInfo.connect(self.update_recording_info)
            self.video_thread.errorOccurred.connect(self.handle_video_error)
            self.video_thread.start()
            
            self.start_button.setText('停止相機')
            self.pause_button.setEnabled(True)
            self.record_button.setEnabled(True)
            self.camera_combo.setEnabled(False)  # 執行時禁用相機選擇
            
            # 更新檢測方法選項
            self.update_detection_methods()
            
        else:
            # 停止相機
            self.video_thread.stop()
            self.video_thread = None
            
            self.start_button.setText('啟動相機')
            self.pause_button.setEnabled(False)
            self.pause_button.setText('暫停')
            self.record_button.setEnabled(False)
            self.record_button.setText('開始錄影')
            self.camera_combo.setEnabled(True)  # 重新啟用相機選擇
            
            # 清空顯示
            self.image_label.clear()
            self.image_label.setText("點擊「啟動相機」開始")
            self.fps_label.setText('FPS: 0.0')
            self.face_count_label.setText('檢測到的臉部: 0')
            self.recording_label.setText('未錄影')
            self.recording_label.setStyleSheet("color: green;")
    
    def handle_video_error(self, error_msg):
        """處理影片錯誤"""
        QMessageBox.critical(self, "錯誤", error_msg)
        if self.video_thread:
            self.toggle_camera()  # 停止相機
    
    def toggle_pause(self):
        """暫停/繼續"""
        if self.video_thread:
            self.video_thread.paused = not self.video_thread.paused
            self.pause_button.setText('繼續' if self.video_thread.paused else '暫停')
    
    def toggle_recording(self):
        """開始/停止錄影"""
        if self.video_thread:
            if self.video_thread.recorder.recording:
                # 停止錄影
                output_file = self.video_thread.recorder.stop_recording()
                self.record_button.setText('開始錄影')
                self.recording_label.setText('未錄影')
                self.recording_label.setStyleSheet("color: green;")
                
                if output_file:
                    QMessageBox.information(self, '錄影完成', f'影片已儲存至:\n{output_file}')
            else:
                # 開始錄影
                if self.video_thread.recorder.start_recording():
                    self.record_button.setText('停止錄影')
                    self.recording_label.setText('錄影中...')
                    self.recording_label.setStyleSheet("color: red;")
                else:
                    QMessageBox.warning(self, '錯誤', '無法開始錄影')
    
    def open_output_folder(self):
        """開啟輸出資料夾"""
        output_dir = "recorded_videos"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':
                os.system(f'open "{output_dir}"')
            else:
                os.system(f'xdg-open "{output_dir}"')
        except Exception as e:
            QMessageBox.warning(self, '錯誤', f'無法開啟資料夾:\n{str(e)}')
    
    def update_detection_methods(self):
        """更新可用的檢測方法"""
        if self.video_thread:
            self.method_combo.clear()
            methods = []
            
            if self.video_thread.tracker.yolo_available:
                methods.append('YOLO')
            if MEDIAPIPE_AVAILABLE:
                methods.append('MediaPipe')
            methods.append('Haar Cascade')
            
            self.method_combo.addItems(methods)
    
    def change_mosaic_style(self, text):
        """改變馬賽克樣式"""
        if self.video_thread:
            style_map = {
                '像素化': 'pixelate',
                '模糊': 'blur',
                '黑色遮擋': 'black'
            }
            self.video_thread.mosaic_style = style_map.get(text, 'pixelate')
    
    def change_mosaic_size(self, value):
        """改變馬賽克大小"""
        self.intensity_value.setText(str(value))
        if self.video_thread:
            self.video_thread.mosaic_size = value
    
    def change_safety_margin(self, value):
        """改變安全邊界"""
        margin = value / 100.0
        self.margin_value.setText(f'{margin:.1f}x')
        if self.video_thread:
            self.video_thread.tracker.safety_margin = margin
    
    def change_detection_method(self, text):
        """改變檢測方法"""
        if self.video_thread:
            method_map = {
                'YOLO': 'yolo',
                'MediaPipe': 'mediapipe',
                'Haar Cascade': 'haar'
            }
            self.video_thread.tracker.detection_method = method_map.get(text, 'haar')
    
    def toggle_child_protection(self, state):
        """切換小孩保護"""
        if self.video_thread:
            self.video_thread.tracker.child_protection_enabled = (state == Qt.Checked)
            if state == Qt.Checked:
                self.video_thread.tracker.face_age_cache.clear()
                self.video_thread.tracker.last_age_detection.clear()
    
    def change_age_threshold(self, value):
        """改變年齡閾值"""
        if self.video_thread:
            self.video_thread.tracker.age_threshold = value
            self.video_thread.tracker.face_age_cache.clear()
            self.video_thread.tracker.last_age_detection.clear()
    
    @pyqtSlot(QImage)
    def update_image(self, image):
        """更新顯示影像"""
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    @pyqtSlot(float)
    def update_fps(self, fps):
        """更新 FPS 顯示"""
        self.fps_label.setText(f'FPS: {fps:.1f}')
    
    @pyqtSlot(int)
    def update_face_count(self, count):
        """更新檢測到的臉部數量"""
        self.face_count_label.setText(f'檢測到的臉部: {count}')
    
    @pyqtSlot(dict)
    def update_recording_info(self, info):
        """更新錄影資訊"""
        text = f"檔案: {info['filename']}\n"
        text += f"錄製時長: {info['duration']:.1f}秒\n"
        text += f"影片時長: {info['estimated_video_duration']:.1f}秒\n"
        text += f"幀數: {info['frame_count']}"
        self.recording_info_label.setText(text)
    
    def closeEvent(self, event):
        """關閉視窗事件"""
        if self.video_thread:
            self.video_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 設定應用程式圖標
    app.setWindowIcon(QIcon())
    
    gui = FaceMosaicGUI()
    gui.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()