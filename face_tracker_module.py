import cv2
import numpy as np
import threading
import time
from collections import deque
import os

# 嘗試導入 DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️  DeepFace 未安裝，小孩保護功能不可用")
    print("   安裝方式: pip install deepface")

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
                model_selection=0, min_detection_confidence=0.4 
            )
        
        # Haar分類器
        self.frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 檢測管理
        self.last_detection_time = time.time()
        self.detection_interval = 0.05  # 每50ms進行一次檢測
        
        # 臉部位置緩存
        self.face_history = deque(maxlen=5)  # 保存最近5幀的檢測結果
        self.smooth_faces = []  # 平滑後的臉部位置
        
        # 性能優化
        self.frame_skip = 0
        self.max_frame_skip = 2
        
        # 安全區域（擴大的遮擋範圍）
        self.safety_margin = 1.3  # 擴大30%
        
        # 檢測方法優先級
        self.detection_method = 'yolo'  # 'yolo', 'mediapipe', 'haar'
        
        # 小孩保護功能
        self.child_protection_enabled = False
        self.age_threshold = 18  # 18歲以下視為小孩
        self.face_age_cache = {}  # 緩存年齡檢測結果
        self.age_detection_interval = 2.0  # 每2秒重新檢測年齡
        self.last_age_detection = {}  # 記錄每張臉的最後檢測時間
        
        # DeepFace 設定
        if DEEPFACE_AVAILABLE:
            self.deepface_backend = 'opencv'  # 使用 opencv 後端以提高速度
            print("✓ DeepFace 已載入，小孩保護功能可用")
        
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
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            
            # 推理
            self.yolo_model.setInput(blob)
            outputs = self.yolo_model.forward()
            
            # 處理輸出
            faces = []
            confidences = []
            
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
                        
                        if w > 20 and h > 20:
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
    
    def smooth_face_positions(self, current_faces):
        """平滑臉部位置以減少抖動"""
        if not current_faces:
            return []
        
        # 將當前檢測結果加入歷史
        self.face_history.append(current_faces)
        
        if len(self.face_history) < 2:
            return current_faces
        
        # 計算平滑位置
        smoothed_faces = []
        
        for i, face in enumerate(current_faces):
            x, y, w, h = face
            
            # 找到歷史中最接近的臉部
            smooth_x, smooth_y, smooth_w, smooth_h = x, y, w, h
            weight_sum = 1.0
            
            for j, history_faces in enumerate(self.face_history):
                if j == len(self.face_history) - 1:  # 跳過當前幀
                    continue
                
                # 權重隨時間遞減
                weight = 0.5 ** (len(self.face_history) - j - 1)
                
                # 找到最接近的臉
                min_dist = float('inf')
                best_match = None
                
                for hist_face in history_faces:
                    hx, hy, hw, hh = hist_face
                    dist = np.sqrt((x - hx)**2 + (y - hy)**2)
                    
                    if dist < min_dist and dist < 100:  # 最大匹配距離
                        min_dist = dist
                        best_match = hist_face
                
                if best_match is not None:
                    hx, hy, hw, hh = best_match
                    smooth_x += hx * weight
                    smooth_y += hy * weight
                    smooth_w += hw * weight
                    smooth_h += hh * weight
                    weight_sum += weight
            
            # 計算加權平均
            smooth_x = int(smooth_x / weight_sum)
            smooth_y = int(smooth_y / weight_sum)
            smooth_w = int(smooth_w / weight_sum)
            smooth_h = int(smooth_h / weight_sum)
            
            smoothed_faces.append((smooth_x, smooth_y, smooth_w, smooth_h))
        
        return smoothed_faces
    
    def analyze_face_age(self, frame, face_rect):
        """使用 DeepFace 分析年齡"""
        if not DEEPFACE_AVAILABLE or not self.child_protection_enabled:
            return None
        
        try:
            x, y, w, h = face_rect
            
            # 稍微擴大區域以確保包含完整的臉
            padding = int(min(w, h) * 0.2)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)
            
            # 提取臉部區域
            face_img = frame[y_start:y_end, x_start:x_end]
            
            # 確保圖像大小足夠
            if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                return None
            
            # 使用 DeepFace 分析年齡
            result = DeepFace.analyze(
                img_path=face_img,
                actions=['age'],
                enforce_detection=False,
                detector_backend=self.deepface_backend,
                silent=True
            )
            
            # 提取年齡
            if isinstance(result, list):
                age = result[0]['age']
            else:
                age = result['age']
            
            return age
            
        except Exception as e:
            # 年齡檢測失敗，默認返回 None
            return None
    
    def get_face_hash(self, face_rect):
        """生成臉部位置的哈希值用於緩存"""
        x, y, w, h = face_rect
        # 使用中心點和大小作為標識
        return f"{x+w//2}_{y+h//2}_{w}_{h}"
    
    def should_apply_mosaic(self, frame, face_rect):
        """判斷是否應該對該臉部應用馬賽克"""
        if not self.child_protection_enabled:
            # 如果未啟用小孩保護，對所有人臉應用馬賽克
            return True
        
        if not DEEPFACE_AVAILABLE:
            # 如果 DeepFace 不可用，為安全起見對所有人臉應用馬賽克
            return True
        
        # 生成臉部標識
        face_hash = self.get_face_hash(face_rect)
        current_time = time.time()
        
        # 檢查緩存
        if face_hash in self.face_age_cache:
            last_time = self.last_age_detection.get(face_hash, 0)
            # 如果在緩存時間內，使用緩存結果
            if current_time - last_time < self.age_detection_interval:
                age = self.face_age_cache[face_hash]
                return age is None or age < self.age_threshold
        
        # 分析年齡
        age = self.analyze_face_age(frame, face_rect)
        
        # 更新緩存
        self.face_age_cache[face_hash] = age
        self.last_age_detection[face_hash] = current_time
        
        # 清理舊緩存（保留最近的50個）
        if len(self.face_age_cache) > 50:
            # 按時間排序，刪除最舊的
            sorted_faces = sorted(self.last_age_detection.items(), key=lambda x: x[1])
            for old_face, _ in sorted_faces[:len(sorted_faces)-50]:
                self.face_age_cache.pop(old_face, None)
                self.last_age_detection.pop(old_face, None)
        
        # 如果年齡檢測失敗或年齡小於閾值，應用馬賽克
        return age is None or age < self.age_threshold
    
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
    
    def update_face_detection(self, frame):
        """更新人臉檢測（無追蹤器版本）"""
        current_time = time.time()
        
        # 檢查是否需要進行新的檢測
        if current_time - self.last_detection_time >= self.detection_interval:
            # 進行檢測
            faces = self.fast_detect_faces(frame)
            self.last_detection_time = current_time
            
            # 平滑臉部位置
            if faces is not None:
                faces = self.smooth_face_positions(faces)
            
            self.smooth_faces = faces
            return faces
        else:
            # 返回上次檢測的結果
            return self.smooth_faces

def apply_smart_mosaic(image, faces, mosaic_size=15, style='pixelate', tracker=None):
    """智能馬賽克應用（支援小孩保護功能）"""
    for i, face in enumerate(faces):
        x, y, w, h = face
        
        # 檢查是否應該應用馬賽克
        if tracker and hasattr(tracker, 'should_apply_mosaic'):
            if not tracker.should_apply_mosaic(image, face):
                # 如果是大人且啟用小孩保護，跳過馬賽克
                continue
        
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
    cap = cv2.VideoCapture(0)
    
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
    
    print("\n=== 高性能人臉馬賽克 (優化版) ===")
    print("✓ YOLOv11n-face 深度學習檢測")
    print("✓ 位置平滑算法")
    print("✓ 小孩保護功能")
    print("✓ 高速處理優化")
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
    print("  c - 切換小孩保護模式")
    print("  a - 調整年齡閾值")
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
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # 人臉檢測
            faces = tracker.update_face_detection(frame)
            
            # 應用馬賽克
            display_frame = apply_smart_mosaic(display_frame, faces, mosaic_size, mosaic_style, tracker)
            
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
            f'Detected Faces: {len(faces)}',
            f'FPS: {current_fps:.1f}',
            f'Processing Time: {avg_frame_time*1000:.1f}ms',
            f'Effect: {mosaic_style} ({mosaic_size})',
            f'Child Protection: {"Enabled" if tracker.child_protection_enabled else "Disabled"}',
        ]
        if tracker.child_protection_enabled and DEEPFACE_AVAILABLE:
            info_texts.append(f'Age Threshold: {tracker.age_threshold} years')
            # 顯示檢測到的年齡資訊
            detected_ages = []
            for face in faces[:3]:  # 只顯示前3個
                face_hash = tracker.get_face_hash(face)
                if face_hash in tracker.face_age_cache:
                    age = tracker.face_age_cache[face_hash]
                    if age is not None:
                        detected_ages.append(int(age))
            if detected_ages:
                info_texts.append(f'Detected Ages: {", ".join(map(str, detected_ages))}')
        
        info_texts.append(f'Status: {"Paused" if paused else "Running"}')
        
        y_offset = 20
        for text in info_texts:
            color = (0, 100, 255) if paused else (0, 255, 0)
            cv2.putText(display_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 18
        
        # 顯示追蹤狀態
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
        elif key == ord('c'):
            # 切換小孩保護模式
            if DEEPFACE_AVAILABLE:
                tracker.child_protection_enabled = not tracker.child_protection_enabled
                print(f"小孩保護模式: {'開啟' if tracker.child_protection_enabled else '關閉'}")
                if tracker.child_protection_enabled:
                    print(f"將只對 {tracker.age_threshold} 歲以下的臉部應用馬賽克")
                # 清空年齡緩存
                tracker.face_age_cache.clear()
                tracker.last_age_detection.clear()
            else:
                print("DeepFace 未安裝，無法使用小孩保護功能")
                print("安裝方式: pip install deepface")
        elif key == ord('a'):
            # 調整年齡閾值
            if DEEPFACE_AVAILABLE:
                print(f"當前年齡閾值: {tracker.age_threshold}")
                print("輸入新的年齡閾值 (建議 16-21):")
                try:
                    # 簡單的輸入處理
                    age_input = ""
                    while True:
                        k = cv2.waitKey(0) & 0xFF
                        if k == 13:  # Enter
                            break
                        elif k == 8:  # Backspace
                            age_input = age_input[:-1]
                        elif 48 <= k <= 57:  # 數字
                            age_input += chr(k)
                        elif k == 27:  # ESC
                            age_input = ""
                            break
                    
                    if age_input:
                        new_age = int(age_input)
                        if 1 <= new_age <= 100:
                            tracker.age_threshold = new_age
                            tracker.face_age_cache.clear()
                            tracker.last_age_detection.clear()
                            print(f"年齡閾值設為: {tracker.age_threshold}")
                        else:
                            print("年齡必須在 1-100 之間")
                except:
                    print("輸入無效")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()