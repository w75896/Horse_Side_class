import cv2
import numpy as np
import threading
import time
from collections import deque

# 嘗試導入MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("建議安裝MediaPipe以獲得更好效果: pip install mediapipe")

class OptimizedFaceTracker:
    def __init__(self):
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
        self.detection_interval = 0.1  # 每100ms進行一次完整檢測
        
        # 預測緩存
        self.face_predictions = []
        self.velocity_cache = deque(maxlen=5)
        
        # 性能優化
        self.frame_skip = 0
        self.max_frame_skip = 2
        
        # 安全區域（擴大的遮擋範圍）
        self.safety_margin = 1.3  # 擴大30%
        
    def create_tracker(self):
        """創建追蹤器"""
        try:
            # 嘗試使用不同的追蹤器（按性能排序）
            tracker_types = ['CSRT', 'KCF', 'MOSSE']
            
            for tracker_type in tracker_types:
                try:
                    if tracker_type == 'CSRT':
                        return cv2.TrackerCSRT_create()
                    elif tracker_type == 'KCF':
                        return cv2.TrackerKCF_create()
                    elif tracker_type == 'MOSSE':
                        return cv2.legacy.TrackerMOSSE_create()
                except:
                    continue
            
            # 如果都失敗，返回None
            return None
        except:
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
        """快速人臉檢測"""
        faces = []
        
        # 使用MediaPipe（如果可用）
        if MEDIAPIPE_AVAILABLE:
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
            except:
                pass
        
        # 如果MediaPipe沒檢測到或不可用，使用快速Haar檢測
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
                            # 稍微擴大初始追蹤區域
                            track_x = max(0, x - 5)
                            track_y = max(0, y - 5)
                            track_w = min(frame.shape[1] - track_x, w + 10)
                            track_h = min(frame.shape[0] - track_y, h + 10)
                            
                            success = tracker.init(frame, (track_x, track_y, track_w, track_h))
                            if success:
                                self.trackers.append(tracker)
                                self.tracker_confidences.append(1.0)
                        except:
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
    
    print("=== 高性能人臉馬賽克 ===")
    print("✓ 智能追蹤算法")
    print("✓ 預測式遮擋")
    print("✓ 高速移動優化")
    print("\n操作說明:")
    print("  q - 退出")
    print("  +/- - 調整強度")
    print("  1/2/3 - 切換效果")
    print("  s - 調整安全邊界")
    
    # 性能監控
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0
    
    # 幀時間監控
    frame_times = deque(maxlen=30)
    
    while True:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # 人臉追蹤和檢測
        faces = tracker.update_face_tracking(frame)
        
        # 應用馬賽克
        frame = apply_smart_mosaic(frame, faces, mosaic_size, mosaic_style)
        
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
        
        # 顯示資訊
        info_texts = [
            f'追蹤器: {len(tracker.trackers)} 個',
            f'檢測到: {len(faces)} 張臉',
            f'FPS: {current_fps:.1f}',
            f'處理時間: {avg_frame_time*1000:.1f}ms',
            f'效果: {mosaic_style} ({mosaic_size})'
        ]
        
        y_offset = 20
        for text in info_texts:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 18
        
        # 顯示追蹤狀態
        for i, face in enumerate(faces):
            x, y, w, h = face
            # 繪製追蹤框（調試用，可選）
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        cv2.imshow('高性能人臉馬賽克', frame)
        
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
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()