import cv2
import numpy as np
import threading
import time
from collections import deque
import os

# 嘗試導入MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("建議安裝MediaPipe以獲得更好效果: pip install mediapipe")

class OptimizedFaceTracker:
    def __init__(self):
        # YOLO 設定
        self.yolo_model = None
        self.yolo_available = False
        self.init_yolo()
        
        # MediaPipe設定
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.4  # 降低閾值增加檢測率
            )
        
        # Haar分類器
        self.frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 追蹤器管理
        self.trackers = []
        self.tracker_confidences = []
        self.last_detection_time = time.time()
        self.detection_interval = 0
        
        # 預測緩存
        self.face_predictions = []
        self.velocity_cache = deque(maxlen=5)
        
        # 性能優化
        self.frame_skip = 0
        self.max_frame_skip = 2
        
        # 安全區域（擴大的遮擋範圍）
        self.safety_margin = 1.3  # 擴大30%
        
        # 檢測方法優先級
        self.detection_method = 'yolo'  # 'yolo', 'mediapipe', 'haar'
        
    def init_yolo(self):
        """初始化 YOLO 模型"""
        try:
            model_path = 'yolov11n-face.onnx'
            if os.path.exists(model_path):
                self.yolo_model = cv2.dnn.readNetFromONNX(model_path)
                self.yolo_available = True
                print("✓ YOLOv11n-face 模型載入成功")
                
                # 設定運算後端
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print("✓ 使用 CUDA 加速")
                else:
                    self.yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    print("✓ 使用 CPU 運算")
            else:
                print(f"⚠️ 找不到 YOLO 模型檔案: {model_path}")
                print("  請確保 yolov11n-face.onnx 在程式同目錄下")
        except Exception as e:
            print(f"⚠️ YOLO 模型載入失敗: {e}")
            self.yolo_available = False
    
    def detect_faces_yolo(self, frame):
        """使用 YOLO 檢測人臉"""
        if not self.yolo_available:
            return []
        
        try:
            height, width = frame.shape[:2]
            
            # 預處理圖像
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), 
                                       swapRB=True, crop=False)
            
            # 推理
            self.yolo_model.setInput(blob)
            outputs = self.yolo_model.forward()
            
            # 處理輸出
            faces = []
            confidences = []
            
            # 打印輸出形狀以便調試
            # print(f"YOLO output shape: {outputs.shape}")
            
            # YOLOv11 輸出格式處理
            # 通常格式是 [1, num_detections, 85] 或 [num_detections, 85]
            if len(outputs.shape) == 3:
                outputs = outputs[0]
            
            # 轉置如果需要（某些版本輸出是 [85, num_detections]）
            if outputs.shape[0] in [5, 6, 85] and outputs.shape[1] > outputs.shape[0]:
                outputs = outputs.T
            
            # 解析檢測結果
            for detection in outputs:
                if len(detection) >= 5:
                    # YOLOv11 格式通常是 [x_center, y_center, width, height, confidence, ...]
                    confidence = float(detection[4])
                    if confidence > 0.3:  # 降低信心度閾值
                        # 獲取邊界框座標（相對於640x640）
                        x_center = detection[0]
                        y_center = detection[1]
                        box_width = detection[2]
                        box_height = detection[3]
                        
                        # 轉換回原始圖像座標
                        scale_x = width / 640.0
                        scale_y = height / 640.0
                        
                        x_center = x_center * scale_x
                        y_center = y_center * scale_y
                        w = box_width * scale_x
                        h = box_height * scale_y
                        
                        # 轉換為左上角座標
                        x = int(x_center - w / 2)
                        y = int(y_center - h / 2)
                        w = int(w)
                        h = int(h)
                        
                        # 確保座標在圖像範圍內
                        x = max(0, x)
                        y = max(0, y)
                        w = min(width - x, w)
                        h = min(height - y, h)
                        
                        if w > 15 and h > 15:
                            faces.append([x, y, w, h])
                            confidences.append(confidence)
            
            # NMS 去除重複檢測
            if len(faces) > 0:
                indices = cv2.dnn.NMSBoxes(faces, confidences, 0.3, 0.4)
                if len(indices) > 0:
                    if isinstance(indices, np.ndarray):
                        indices = indices.flatten()
                    return [tuple(faces[i]) for i in indices]
            
            return []
            
        except Exception as e:
            print(f"YOLO 檢測錯誤: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def create_tracker(self):
        """創建追蹤器 - 針對 OpenCV 4.8.1 優化"""
        try:
            # 獲取 OpenCV 版本
            cv_version = cv2.__version__.split('.')
            major_version = int(cv_version[0])
            minor_version = int(cv_version[1])
            
            # OpenCV 4.5+ 需要使用 legacy 追蹤器
            if major_version >= 4 and minor_version >= 5:
                # 使用 legacy 追蹤器（按性能排序）
                tracker_creators = [
                    ('CSRT (legacy)', lambda: cv2.legacy.TrackerCSRT_create()),
                    ('KCF (legacy)', lambda: cv2.legacy.TrackerKCF_create()),
                    ('MOSSE (legacy)', lambda: cv2.legacy.TrackerMOSSE_create()),
                    ('MIL (legacy)', lambda: cv2.legacy.TrackerMIL_create()),
                    ('MEDIANFLOW (legacy)', lambda: cv2.legacy.TrackerMedianFlow_create()),
                    ('TLD (legacy)', lambda: cv2.legacy.TrackerTLD_create()),
                ]
            else:
                # 舊版本使用標準追蹤器
                tracker_creators = [
                    ('CSRT', lambda: cv2.TrackerCSRT_create()),
                    ('KCF', lambda: cv2.TrackerKCF_create()),
                    ('MIL', lambda: cv2.TrackerMIL_create()),
                ]
            
            # 嘗試創建追蹤器
            for tracker_name, creator_func in tracker_creators:
                try:
                    tracker = creator_func()
                    # print(f"成功創建 {tracker_name} 追蹤器")
                    return tracker
                except Exception as e:
                    continue
            
            print("警告：無法創建任何追蹤器")
            print(f"OpenCV 版本: {cv2.__version__}")
            return None
            
        except Exception as e:
            print(f"創建追蹤器時發生錯誤: {e}")
            return None
    
    def predict_face_position(self, face_rect, velocity):
        """根據速度預測人臉位置"""
        x, y, w, h = face_rect
        vx, vy = velocity
        
        # 預測下一幀的位置
        predicted_x = int(x + vx * 2)  # 預測2幀後的位置
        predicted_y = int(y + vy * 2)
        
        return (predicted_x, predicted_y, w, h)
    
    def calculate_velocity(self, current_faces, previous_faces):
        """計算人臉移動速度"""
        velocities = []
        
        for curr_face in current_faces:
            cx, cy, cw, ch = curr_face
            curr_center = (cx + cw//2, cy + ch//2)
            
            best_match = None
            min_distance = float('inf')
            
            # 找到最匹配的前一幀人臉
            for prev_face in previous_faces:
                px, py, pw, ph = prev_face
                prev_center = (px + pw//2, py + ph//2)
                
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                 (curr_center[1] - prev_center[1])**2)
                
                if distance < min_distance and distance < 100:  # 最大匹配距離
                    min_distance = distance
                    best_match = prev_center
            
            if best_match:
                vx = curr_center[0] - best_match[0]
                vy = curr_center[1] - best_match[1]
                velocities.append((vx, vy))
            else:
                velocities.append((0, 0))
        
        return velocities
    
    def expand_face_region(self, face_rect, frame_shape, margin=None):
        """擴大人臉區域以提供安全邊界"""
        if margin is None:
            margin = self.safety_margin
            
        x, y, w, h = face_rect
        frame_h, frame_w = frame_shape[:2]
        
        # 計算擴展尺寸
        new_w = int(w * margin)
        new_h = int(h * margin)
        
        # 計算新的左上角位置（保持中心點）
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)
        
        # 確保不超出框架邊界
        new_w = min(new_w, frame_w - new_x)
        new_h = min(new_h, frame_h - new_y)
        
        return (new_x, new_y, new_w, new_h)
    
    def fast_detect_faces(self, frame):
        """快速人臉檢測（支援多種方法）"""
        faces = []
        
        # 根據優先級使用不同的檢測方法
        if self.detection_method == 'yolo' and self.yolo_available:
            faces = self.detect_faces_yolo(frame)
            if len(faces) > 0:
                return faces
        
        # 使用MediaPipe（如果可用）
        if MEDIAPIPE_AVAILABLE and (self.detection_method == 'mediapipe' or len(faces) == 0):
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)
                
                if results.detections:
                    h, w = frame.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = max(0, int(bbox.xmin * w))
                        y = max(0, int(bbox.ymin * h))
                        width = min(w - x, int(bbox.width * w))
                        height = min(h - y, int(bbox.height * h))
                        
                        if width > 15 and height > 15:
                            faces.append((x, y, width, height))
                    
                    if len(faces) > 0:
                        return faces
            except:
                pass
        
        # 如果還是沒檢測到，使用快速Haar檢測
        if len(faces) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 使用較寬鬆的參數進行快速檢測
            detected = self.frontal_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=2, minSize=(20, 20)
            )
            faces.extend(detected)
        
        return faces
    
    def update_trackers(self, frame):
        """更新追蹤器"""
        updated_faces = []
        valid_trackers = []
        valid_confidences = []
        
        for i, tracker in enumerate(self.trackers):
            try:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    # 檢查邊界
                    if (x >= 0 and y >= 0 and x + w <= frame.shape[1] and 
                        y + h <= frame.shape[0] and w > 10 and h > 10):
                        updated_faces.append((x, y, w, h))
                        valid_trackers.append(tracker)
                        valid_confidences.append(self.tracker_confidences[i] * 0.95)  # 逐漸降低信心度
                    else:
                        # 追蹤失敗，移除追蹤器
                        pass
                else:
                    # 追蹤失敗
                    pass
            except:
                # 追蹤器出錯
                pass
        
        self.trackers = valid_trackers
        self.tracker_confidences = valid_confidences
        
        return updated_faces
    
    def merge_detections_and_tracking(self, detected_faces, tracked_faces):
        """合併檢測和追蹤結果"""
        all_faces = []
        
        # 添加檢測到的人臉
        for face in detected_faces:
            all_faces.append((face, 1.0, 'detected'))
        
        # 添加追蹤的人臉（如果沒有重疊的檢測結果）
        for i, tracked_face in enumerate(tracked_faces):
            tx, ty, tw, th = tracked_face
            is_duplicate = False
            
            for detected_face in detected_faces:
                dx, dy, dw, dh = detected_face
                
                # 計算重疊
                overlap_x = max(0, min(tx + tw, dx + dw) - max(tx, dx))
                overlap_y = max(0, min(ty + th, dy + dh) - max(ty, dy))
                overlap_area = overlap_x * overlap_y
                
                if overlap_area > 0.3 * min(tw * th, dw * dh):
                    is_duplicate = True
                    break
            
            if not is_duplicate and self.tracker_confidences[i] > 0.3:
                all_faces.append((tracked_face, self.tracker_confidences[i], 'tracked'))
        
        # 只返回人臉區域
        return [face for face, conf, source in all_faces if conf > 0.2]
    
    def update_face_tracking(self, frame):
        """主要的人臉追蹤更新方法"""
        current_time = time.time()
        faces = []
        
        # 更新現有追蹤器
        tracked_faces = self.update_trackers(frame)
        
        # 決定是否進行新的檢測
        should_detect = (current_time - self.last_detection_time > self.detection_interval or 
                        len(self.trackers) == 0)
        
        if should_detect:
            # 進行新的檢測
            detected_faces = self.fast_detect_faces(frame)
            self.last_detection_time = current_time
            
            # 為新檢測到的人臉創建追蹤器
            for face in detected_faces:
                x, y, w, h = face
                
                # 檢查是否已經有追蹤器追蹤類似區域
                is_new_face = True
                for tracked_face in tracked_faces:
                    tx, ty, tw, th = tracked_face
                    distance = np.sqrt((x - tx)**2 + (y - ty)**2)
                    if distance < 50:  # 如果距離很近，認為是同一張臉
                        is_new_face = False
                        break
                
                if is_new_face:
                    tracker = self.create_tracker()
                    if tracker:
                        try:
                            # 確保座標是整數
                            x, y, w, h = int(x), int(y), int(w), int(h)
                            
                            # 確保寬高大於最小值
                            if w < 20 or h < 20:
                                continue
                            
                            # 不要擴大初始區域，使用原始檢測框
                            bbox = (x, y, w, h)
                            
                            # 初始化追蹤器
                            success = tracker.init(frame, bbox)
                            
                            if success:
                                self.trackers.append(tracker)
                                self.tracker_confidences.append(1.0)
                                # print(f"✓ 追蹤器初始化成功: {bbox}")
                            else:
                                # Legacy 追蹤器可能需要不同的初始化方式
                                pass
                        except Exception as e:
                            # print(f"追蹤器初始化錯誤: {e}")
                            pass
                    else:
                        # 如果無法創建追蹤器，仍然返回檢測結果
                        pass
            
            # 合併檢測和追蹤結果
            faces = self.merge_detections_and_tracking(detected_faces, tracked_faces)
        else:
            # 只使用追蹤結果
            faces = tracked_faces
        
        # 清理低信心度的追蹤器
        self.trackers = [t for i, t in enumerate(self.trackers) if self.tracker_confidences[i] > 0.2]
        self.tracker_confidences = [c for c in self.tracker_confidences if c > 0.2]
        
        return faces

def apply_smart_mosaic(image, faces, mosaic_size=15, style='pixelate'):
    """智能馬賽克應用（帶預測）"""
    for face in faces:
        x, y, w, h = face
        
        # 擴大保護區域
        padding = max(10, min(w, h) // 8)
        safe_x = max(0, x - padding)
        safe_y = max(0, y - padding)
        safe_w = min(image.shape[1] - safe_x, w + 2 * padding)
        safe_h = min(image.shape[0] - safe_y, h + 2 * padding)
        
        if safe_w <= 5 or safe_h <= 5:
            continue
        
        face_region = image[safe_y:safe_y+safe_h, safe_x:safe_x+safe_w].copy()
        
        if style == 'pixelate':
            block_size = max(2, min(mosaic_size, min(safe_w, safe_h) // 8))
            small = cv2.resize(face_region, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (safe_w, safe_h), interpolation=cv2.INTER_NEAREST)
        elif style == 'blur':
            kernel_size = max(5, mosaic_size * 2 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            mosaic = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
        else:  # 'black'
            mosaic = np.zeros_like(face_region)
        
        # 應用漸變邊緣以減少突兀感
        if style != 'black':
            mask = np.ones((safe_h, safe_w, 3), dtype=np.float32)
            border_size = min(10, min(safe_w, safe_h) // 10)
            
            # 創建漸變遮罩
            for i in range(border_size):
                alpha = i / border_size
                mask[i, :] *= alpha
                mask[-(i+1), :] *= alpha
                mask[:, i] *= alpha
                mask[:, -(i+1)] *= alpha
            
            # 混合原圖和馬賽克
            mosaic = (mosaic * mask + face_region * (1 - mask)).astype(np.uint8)
        
        image[safe_y:safe_y+safe_h, safe_x:safe_x+safe_w] = mosaic
    
    return image

def main():
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("錯誤：無法開啟攝影機")
        return
    
    # 優化攝影機設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    tracker = OptimizedFaceTracker()
    mosaic_size = 15
    mosaic_style = 'pixelate'
    
    print("\n=== 高性能人臉馬賽克 (YOLOv11n 增強版) ===")
    print("✓ YOLOv11n-face 深度學習檢測")
    print("✓ 智能追蹤算法")
    print("✓ 預測式遮擋")
    print("✓ 高速移動優化")
    print("\n檢測方法:")
    if tracker.yolo_available:
        print("  ► YOLO (主要)")
    if MEDIAPIPE_AVAILABLE:
        print("  ► MediaPipe (備用)")
    print("  ► Haar Cascade (後備)")
    
    print("\n操作說明:")
    print("  q - 退出")
    print("  +/- - 調整強度")
    print("  1/2/3 - 切換效果")
    print("  s - 調整安全邊界")
    print("  d - 切換檢測方法")
    print("  SPACE - 暫停/繼續")
    
    # 性能監控
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0
    
    # 幀時間監控
    frame_times = deque(maxlen=30)
    
    # 暫停狀態
    paused = False
    
    while True:
        if not paused:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # 人臉追蹤和檢測
            faces = tracker.update_face_tracking(frame)
            
            # 應用馬賽克
            display_frame = apply_smart_mosaic(display_frame, faces, mosaic_size, mosaic_style)
            
            # 性能統計
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            
            fps_counter += 1
            if fps_counter >= 15:
                current_fps = 15 / (time.time() - fps_timer)
                fps_timer = time.time()
                fps_counter = 0
            
            avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        else:
            display_frame = frame.copy() if 'frame' in locals() else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 顯示資訊
        info_texts = [
            f'Detection Method: {tracker.detection_method.upper()}',
            f'Trackers: {len(tracker.trackers)}',
            f'Faces Detected: {len(faces)}',
            f'FPS: {current_fps:.1f}',
            f'Processing Time: {avg_frame_time*1000:.1f}ms',
            f'Effect: {mosaic_style} ({mosaic_size})',
            f'Status: {"Paused" if paused else "Running"}'
        ]
        
        y_offset = 20
        for text in info_texts:
            color = (0, 100, 255) if paused else (0, 255, 0)
            cv2.putText(display_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 18
        
        # 顯示追蹤狀態（調試用，可選）
        # for i, face in enumerate(faces):
        #     x, y, w, h = face
        #     cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        cv2.imshow('高性能人臉馬賽克 - YOLOv11n', display_frame)
        
        # 按鍵處理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            mosaic_size = max(3, mosaic_size - 2)
            print(f"馬賽克強度: {mosaic_size}")
        elif key == ord('-'):
            mosaic_size = min(50, mosaic_size + 2)
            print(f"馬賽克強度: {mosaic_size}")
        elif key == ord('1'):
            mosaic_style = 'pixelate'
            print("像素化馬賽克")
        elif key == ord('2'):
            mosaic_style = 'blur'
            print("模糊效果")
        elif key == ord('3'):
            mosaic_style = 'black'
            print("黑色遮擋")
        elif key == ord('s'):
            tracker.safety_margin = 1.5 if tracker.safety_margin < 1.4 else 1.2
            print(f"安全邊界: {tracker.safety_margin:.1f}x")
        elif key == ord('d'):
            # 切換檢測方法
            methods = []
            if tracker.yolo_available:
                methods.append('yolo')
            if MEDIAPIPE_AVAILABLE:
                methods.append('mediapipe')
            methods.append('haar')  # Haar 總是可用
            
            if len(methods) > 1:
                current_idx = methods.index(tracker.detection_method) if tracker.detection_method in methods else 0
                next_idx = (current_idx + 1) % len(methods)
                tracker.detection_method = methods[next_idx]
                print(f"切換到 {tracker.detection_method.upper()} 檢測方法")
            else:
                print(f"只有 {tracker.detection_method.upper()} 檢測方法可用")
        elif key == ord(' '):
            paused = not paused
            print("暫停" if paused else "繼續")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()