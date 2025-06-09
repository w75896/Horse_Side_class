import cv2
import numpy as np
import threading
import time
from collections import deque
import os
from datetime import datetime

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✓ DeepFace 載入成功")
except ImportError as e:
    DEEPFACE_AVAILABLE = False
    print("✗ DeepFace 載入失敗 (小孩保護功能將不可用)")
    print(f"  錯誤: {str(e)}")
except Exception as e:
    DEEPFACE_AVAILABLE = False
    print("✗ DeepFace 載入時發生錯誤")
    print(f"  錯誤類型: {type(e).__name__}")
    print(f"  錯誤訊息: {str(e)}")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe 載入成功")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print("✗ MediaPipe 載入失敗 (人臉檢測功能將不可用)")
    print(f"  錯誤: {str(e)}")

class VideoRecorder:
    """影片錄製管理器"""
    def __init__(self):
        self.recording = False
        self.video_writer = None
        self.output_filename = None
        self.start_time = None
        self.frame_count = 0
        
        self.target_fps = 30 
        self.actual_fps = 30 
        self.frame_size = (640, 480)
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        

        self.fps_history = deque(maxlen=30)
        self.last_fps_update = time.time()
        self.fps_update_interval = 1.0 
        
        self.output_dir = "recorded_videos"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def update_fps(self, current_fps):
        """更新實際FPS"""
        current_time = time.time()
        if current_fps > 0:
            self.fps_history.append(current_fps)

        if current_time - self.last_fps_update >= self.fps_update_interval:
            if len(self.fps_history) > 0:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                self.actual_fps = max(5, min(60, avg_fps))
            self.last_fps_update = current_time
    
    def start_recording(self, frame_size=None, current_fps=None):
        """開始錄製"""
        if self.recording:
            print(" 已在錄製中")
            return False
        
        if frame_size:
            self.frame_size = frame_size
        
        if current_fps and current_fps > 0:
            self.actual_fps = max(5, min(60, current_fps))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filename = os.path.join(self.output_dir, f"mosaic_video_{timestamp}.mp4")
   
        self.video_writer = cv2.VideoWriter(
            self.output_filename,
            self.codec,
            self.actual_fps, 
            self.frame_size
        )
        
        if not self.video_writer.isOpened():
            print("無法初始化影片寫入器")
            return False
        
        self.recording = True
        self.start_time = time.time()
        self.frame_count = 0
        self.fps_history.clear() 
        
        print(f" 開始錄製: {self.output_filename}")
        print(f"   設定FPS: {self.actual_fps:.1f}")
        return True
    
    def write_frame(self, frame):
        """寫入一幀"""
        if not self.recording or self.video_writer is None:
            return
        
        if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
            frame = cv2.resize(frame, self.frame_size)
        
        self.video_writer.write(frame)
        self.frame_count += 1
    
    def stop_recording(self):
        """停止錄製"""
        if not self.recording:
            print("目前未在錄製")
            return None
        
        self.recording = False
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        duration = time.time() - self.start_time if self.start_time else 0
        estimated_video_duration = self.frame_count / self.actual_fps if self.actual_fps > 0 else 0
        
        print(f"錄製完成!")
        print(f"   檔案: {self.output_filename}")
        print(f"   實際錄製時長: {duration:.1f} 秒")
        print(f"   影片播放時長: {estimated_video_duration:.1f} 秒")
        print(f"   總幀數: {self.frame_count}")
        print(f"   錄製FPS: {self.frame_count/duration:.1f}" if duration > 0 else "")
        print(f"   影片FPS: {self.actual_fps:.1f}")
        
        return self.output_filename
    
    def get_recording_info(self):
        """獲取錄製資訊"""
        if not self.recording:
            return None
        
        duration = time.time() - self.start_time if self.start_time else 0
        estimated_video_duration = self.frame_count / self.actual_fps if self.actual_fps > 0 else 0
        
        return {
            'filename': os.path.basename(self.output_filename),
            'duration': duration,
            'frame_count': self.frame_count,
            'fps': self.frame_count / duration if duration > 0 else 0,
            'actual_fps': self.actual_fps,
            'estimated_video_duration': estimated_video_duration
        }

class OptimizedFaceTracker:
    def __init__(self):
        self.yolo_model = None
        self.yolo_available = False
        self.init_yolo()
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.4 
            )
        
        self.frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.last_detection_time = time.time()
        self.detection_interval = 0.05 
        
        self.face_history = deque(maxlen=5)  
        self.smooth_faces = [] 
        
        self.frame_skip = 0
        self.max_frame_skip = 2
        
        self.safety_margin = 1.3 
        
        self.detection_method = 'yolo'  
        
        self.child_protection_enabled = False
        self.age_threshold = 18  
        self.face_age_cache = {} 
        self.age_detection_interval = 1.0  
        self.last_age_detection = {} 
        
        if DEEPFACE_AVAILABLE:
            self.deepface_backend = 'opencv'  
            print(" DeepFace 已載入，小孩保護功能可用")
        
    def init_yolo(self):
        """初始化 YOLO 模型"""
        try:
            model_path = 'yolov11n-face.onnx'
            if os.path.exists(model_path):
                self.yolo_model = cv2.dnn.readNetFromONNX(model_path)
                self.yolo_available = True
                print(" YOLOv11n-face 模型載入成功")
                
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print(" 使用 CUDA 加速")
                else:
                    self.yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    print(" 使用 CPU 運算")
            else:
                print(f"找不到 YOLO 模型檔案: {model_path}")
                print("  請確保 yolov11n-face.onnx 在程式同目錄下")
        except Exception as e:
            print(f"YOLO 模型載入失敗: {e}")
            self.yolo_available = False
    
    def detect_faces_yolo(self, frame):
        """使用 YOLO 檢測人臉"""
        if not self.yolo_available:
            return []
        
        try:
            height, width = frame.shape[:2]
            
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            
            self.yolo_model.setInput(blob)
            outputs = self.yolo_model.forward()
            
            faces = []
            confidences = []
            
            if len(outputs.shape) == 3:
                outputs = outputs[0]
            
            if outputs.shape[0] in [5, 6, 85] and outputs.shape[1] > outputs.shape[0]:
                outputs = outputs.T
            
            for detection in outputs:
                if len(detection) >= 5:
                    confidence = float(detection[4])
                    if confidence > 0.3: 
                        x_center = detection[0]
                        y_center = detection[1]
                        box_width = detection[2]
                        box_height = detection[3]
                        
                        scale_x = width / 640.0
                        scale_y = height / 640.0
                        
                        x_center = x_center * scale_x
                        y_center = y_center * scale_y
                        w = box_width * scale_x
                        h = box_height * scale_y
                        
                        x = int(x_center - w / 2)
                        y = int(y_center - h / 2)
                        w = int(w)
                        h = int(h)
                        
                        x = max(0, x)
                        y = max(0, y)
                        w = min(width - x, w)
                        h = min(height - y, h)
                        
                        if w > 20 and h > 20:
                            faces.append([x, y, w, h])
                            confidences.append(confidence)
            
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
        
        new_w = int(w * margin)
        new_h = int(h * margin)
        
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)
        
        new_w = min(new_w, frame_w - new_x)
        new_h = min(new_h, frame_h - new_y)
        
        return (new_x, new_y, new_w, new_h)
    
    def smooth_face_positions(self, current_faces):
        """平滑臉部位置以減少抖動"""
        if not current_faces:
            return []
        
        self.face_history.append(current_faces)
        
        if len(self.face_history) < 2:
            return current_faces
        
        smoothed_faces = []
        
        for i, face in enumerate(current_faces):
            x, y, w, h = face
            
            smooth_x, smooth_y, smooth_w, smooth_h = x, y, w, h
            weight_sum = 1.0
            
            for j, history_faces in enumerate(self.face_history):
                if j == len(self.face_history) - 1:  
                    continue
                
                weight = 0.5 ** (len(self.face_history) - j - 1)
                
                min_dist = float('inf')
                best_match = None
                
                for hist_face in history_faces:
                    hx, hy, hw, hh = hist_face
                    dist = np.sqrt((x - hx)**2 + (y - hy)**2)
                    
                    if dist < min_dist and dist < 100: 
                        min_dist = dist
                        best_match = hist_face
                
                if best_match is not None:
                    hx, hy, hw, hh = best_match
                    smooth_x += hx * weight
                    smooth_y += hy * weight
                    smooth_w += hw * weight
                    smooth_h += hh * weight
                    weight_sum += weight
            
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
            
            padding = int(min(w, h) * 0.2)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)
            
            face_img = frame[y_start:y_end, x_start:x_end]
            
            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                return None
            
            result = DeepFace.analyze(
                img_path=face_img,
                actions=['age'],
                enforce_detection=False,
                detector_backend=self.deepface_backend,
                silent=True
            )
            
            if isinstance(result, list):
                age = result[0]['age']
            else:
                age = result['age']
            
            return age
            
        except Exception as e:
            return None
    
    def get_face_hash(self, face_rect):
        """生成臉部位置的哈希值用於緩存"""
        x, y, w, h = face_rect
        return f"{x+w//2}_{y+h//2}_{w}_{h}"
    
    def should_apply_mosaic(self, frame, face_rect):
        """判斷是否應該對該臉部應用馬賽克"""
        if not self.child_protection_enabled:
            return True
        
        if not DEEPFACE_AVAILABLE:
            return True
        
        face_hash = self.get_face_hash(face_rect)
        current_time = time.time()
        
        if face_hash in self.face_age_cache:
            last_time = self.last_age_detection.get(face_hash, 0)
            if current_time - last_time < self.age_detection_interval:
                age = self.face_age_cache[face_hash]
                return age is None or age < self.age_threshold
        
        age = self.analyze_face_age(frame, face_rect)
        
        self.face_age_cache[face_hash] = age
        self.last_age_detection[face_hash] = current_time
        
        if len(self.face_age_cache) > 50:
            sorted_faces = sorted(self.last_age_detection.items(), key=lambda x: x[1])
            for old_face, _ in sorted_faces[:len(sorted_faces)-50]:
                self.face_age_cache.pop(old_face, None)
                self.last_age_detection.pop(old_face, None)
        
        return age is None or age < self.age_threshold
    
    def fast_detect_faces(self, frame):
        """快速人臉檢測（支援多種方法）"""
        faces = []
        
        if self.detection_method == 'yolo' and self.yolo_available:
            faces = self.detect_faces_yolo(frame)
            if len(faces) > 0:
                return faces
        
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
        
        if len(faces) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.frontal_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=2, minSize=(20, 20)
            )
            faces.extend(detected)
        
        return faces
    
    def update_face_detection(self, frame):
        """更新人臉檢測"""
        current_time = time.time()
        
        if current_time - self.last_detection_time >= self.detection_interval:
            faces = self.fast_detect_faces(frame)
            self.last_detection_time = current_time
            
            if faces is not None:
                faces = self.smooth_face_positions(faces)
            
            self.smooth_faces = faces
            return faces
        else:
            return self.smooth_faces

def apply_smart_mosaic(image, faces, mosaic_size=15, style='pixelate', tracker=None):
    """智能馬賽克應用（支援小孩保護功能）"""
    for i, face in enumerate(faces):
        x, y, w, h = face
        
        if tracker and hasattr(tracker, 'should_apply_mosaic'):
            if not tracker.should_apply_mosaic(image, face):
                continue
        
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
        else: 
            mosaic = np.zeros_like(face_region)
        
        if style != 'black':
            mask = np.ones((safe_h, safe_w, 3), dtype=np.float32)
            border_size = min(10, min(safe_w, safe_h) // 10)
            
            for i in range(border_size):
                alpha = i / border_size
                mask[i, :] *= alpha
                mask[-(i+1), :] *= alpha
                mask[:, i] *= alpha
                mask[:, -(i+1)] *= alpha
            
            mosaic = (mosaic * mask + face_region * (1 - mask)).astype(np.uint8)
        
        image[safe_y:safe_y+safe_h, safe_x:safe_x+safe_w] = mosaic
    
    return image
