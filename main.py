import cv2
import numpy as np
import sys

# 嘗試導入MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe 可用 - 將使用高精度人臉檢測")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe 未安裝 - 將使用增強版Haar檢測")
    print("建議安裝MediaPipe以獲得更好的檢測效果: pip install mediapipe")

class EnhancedFaceDetector:
    def __init__(self):
        # MediaPipe人臉檢測
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.3
            )
            print("使用MediaPipe人臉檢測")
        
        # Haar級聯分類器（作為備用或補充）
        self.frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # 檢測歷史（用於穩定檢測）
        self.detection_history = []
        self.history_size = 5
        
    def detect_with_mediapipe(self, frame):
        """使用MediaPipe檢測人臉"""
        if not MEDIAPIPE_AVAILABLE:
            return []
        
        try:
            # 轉換BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # 轉換為絕對座標
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # 確保座標在有效範圍內
                    x = max(0, x)
                    y = max(0, y)
                    width = min(w - x, width)
                    height = min(h - y, height)
                    
                    if width > 20 and height > 20:  # 過濾太小的檢測
                        faces.append((x, y, width, height))
            
            return faces
        except Exception as e:
            print(f"MediaPipe檢測錯誤: {e}")
            return []
    
    def detect_with_haar_enhanced(self, frame):
        """增強版Haar檢測（多角度、多尺度）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        
        # 多尺度參數組合
        scale_factors = [1.05, 1.1, 1.2]
        min_neighbors_list = [3, 4, 5]
        
        # 正面人臉檢測（多參數）
        for scale in scale_factors:
            for min_neighbors in min_neighbors_list:
                detected = self.frontal_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale, 
                    minNeighbors=min_neighbors, 
                    minSize=(25, 25),
                    maxSize=(300, 300)
                )
                faces.extend(detected)
        
        # 側面人臉檢測
        for scale in [1.05, 1.1]:
            # 左側面
            profile_faces = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=scale, minNeighbors=3, minSize=(25, 25)
            )
            faces.extend(profile_faces)
            
            # 右側面（翻轉檢測）
            flipped_gray = cv2.flip(gray, 1)
            flipped_faces = self.profile_cascade.detectMultiScale(
                flipped_gray, scaleFactor=scale, minNeighbors=3, minSize=(25, 25)
            )
            
            # 調整翻轉後的座標
            frame_width = frame.shape[1]
            for (x, y, w, h) in flipped_faces:
                adjusted_x = frame_width - x - w
                faces.append((adjusted_x, y, w, h))
        
        return faces
    
    def filter_overlapping_faces(self, faces, overlap_threshold=0.3):
        """過濾重疊的人臉檢測"""
        if len(faces) <= 1:
            return faces
        
        # 按面積排序（大的優先）
        faces_with_area = [(face, face[2] * face[3]) for face in faces]
        faces_with_area.sort(key=lambda x: x[1], reverse=True)
        
        filtered_faces = []
        
        for current_face, _ in faces_with_area:
            x1, y1, w1, h1 = current_face
            is_duplicate = False
            
            for existing_face in filtered_faces:
                x2, y2, w2, h2 = existing_face
                
                # 計算重疊面積
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                # 計算較小矩形的面積
                area1 = w1 * h1
                area2 = w2 * h2
                smaller_area = min(area1, area2)
                
                # 如果重疊面積超過閾值，認為是重複
                if overlap_area > overlap_threshold * smaller_area:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_faces.append(current_face)
        
        return filtered_faces
    
    def stabilize_detection(self, faces):
        """穩定檢測結果"""
        self.detection_history.append(faces)
        
        # 保持歷史記錄大小
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # 如果歷史記錄不足，直接返回當前結果
        if len(self.detection_history) < 3:
            return faces
        
        # 統計每個位置的檢測頻率
        stable_faces = []
        
        for face in faces:
            x, y, w, h = face
            confidence = 0
            
            # 檢查歷史記錄中相似的檢測
            for history_faces in self.detection_history[-3:]:  # 檢查最近3幀
                for hist_face in history_faces:
                    hx, hy, hw, hh = hist_face
                    
                    # 計算位置相似度
                    distance = np.sqrt((x - hx)**2 + (y - hy)**2)
                    size_diff = abs(w - hw) + abs(h - hh)
                    
                    if distance < 30 and size_diff < 20:  # 相似閾值
                        confidence += 1
            
            # 只保留在多幀中都出現的人臉
            if confidence >= 2:
                stable_faces.append(face)
        
        return stable_faces
    
    def detect_faces(self, frame):
        """主要人臉檢測方法"""
        faces = []
        
        # 優先使用MediaPipe
        if MEDIAPIPE_AVAILABLE:
            mp_faces = self.detect_with_mediapipe(frame)
            faces.extend(mp_faces)
        
        # 如果MediaPipe檢測數量少，補充Haar檢測
        if len(faces) == 0 or not MEDIAPIPE_AVAILABLE:
            haar_faces = self.detect_with_haar_enhanced(frame)
            faces.extend(haar_faces)
        
        # 過濾重疊
        faces = self.filter_overlapping_faces(faces)
        
        # 穩定檢測
        faces = self.stabilize_detection(faces)
        
        return faces

def apply_advanced_mosaic(image, x, y, w, h, mosaic_size=15, style='pixelate'):
    """高級馬賽克效果"""
    # 擴展區域以確保完全覆蓋
    padding = max(5, min(w, h) // 20)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    if w <= 5 or h <= 5:
        return image
    
    face_region = image[y:y+h, x:x+w].copy()
    
    if style == 'pixelate':
        # 像素化馬賽克
        block_size = max(1, min(mosaic_size, min(w, h) // 6))
        small = cv2.resize(face_region, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    elif style == 'blur':
        # 模糊效果
        kernel_size = max(3, mosaic_size * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mosaic = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
    
    else:  # 'black'
        # 黑色遮擋
        mosaic = np.zeros_like(face_region)
    
    image[y:y+h, x:x+w] = mosaic
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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 減少延遲
    
    detector = EnhancedFaceDetector()
    mosaic_size = 15
    mosaic_style = 'pixelate'  # 'pixelate', 'blur', 'black'
    
    print("=== 超級人臉馬賽克程式 ===")
    if MEDIAPIPE_AVAILABLE:
        print("✓ 使用MediaPipe高精度檢測")
    print("✓ 多角度檢測支援")
    print("✓ 智能穩定算法")
    print("\n操作說明:")
    print("  q - 退出程式")
    print("  + - 增加馬賽克強度")
    print("  - - 減少馬賽克強度") 
    print("  1 - 像素化馬賽克")
    print("  2 - 模糊效果")
    print("  3 - 黑色遮擋")
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機畫面")
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # 檢測人臉
        faces = detector.detect_faces(frame)
        
        # 應用馬賽克
        for (x, y, w, h) in faces:
            frame = apply_advanced_mosaic(frame, x, y, w, h, mosaic_size, mosaic_style)
        
        # 計算FPS
        fps_counter += 1
        if fps_counter >= 30:
            fps_end_time = cv2.getTickCount()
            fps = 30 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
            fps_start_time = fps_end_time
            fps_counter = 0
        else:
            fps = 0
        
        # 顯示資訊
        info_texts = [
            f'檢測器: {"MediaPipe+Haar" if MEDIAPIPE_AVAILABLE else "增強Haar"}',
            f'馬賽克: {mosaic_style} ({mosaic_size})',
            f'人臉: {len(faces)}',
            f'FPS: {fps:.1f}' if fps > 0 else 'FPS: 計算中...'
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(frame, text, (10, 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('超級人臉馬賽克', frame)
        
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
            print("切換到像素化馬賽克")
        elif key == ord('2'):
            mosaic_style = 'blur'
            print("切換到模糊效果")
        elif key == ord('3'):
            mosaic_style = 'black'
            print("切換到黑色遮擋")
    
    cap.release()
    cv2.destroyAllWindows()
    print("程式結束")

if __name__ == "__main__":
    main()