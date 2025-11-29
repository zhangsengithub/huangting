#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import time

class CameraDetector:
    def __init__(self, model_path='yolov8n.pt', camera_id=0):
        """
        初始化检测器
        :param model_path: 模型路径
        :param camera_id: 摄像头ID (0通常是默认摄像头)
        """
        self.model = YOLO(model_path)
        self.camera_id = camera_id
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def open_camera(self):
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            print(f"错误：无法打开摄像头 {self.camera_id}")
            return False
        return True
    
    def calculate_fps(self):
        """计算FPS"""
        self.frame_count += 1
        if self.frame_count >= 30:
            end_time = time.time()
            self.fps = self.frame_count / (end_time - self.start_time)
            self.frame_count = 0
            self.start_time = end_time
    
    def draw_detections(self, frame, results):
        """在帧上绘制检测结果"""
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取坐标和置信度
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # 绘制边界框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 添加标签和置信度
                label = f"{result.names[cls]}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """主运行循环"""
        if not self.open_camera():
            return
        
        print("开始摄像头检测，按 'q' 退出，按 's' 保存当前帧")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            # 进行目标检测
            results = self.model(frame, verbose=False)
            
            # 绘制检测结果
            frame_with_detections = self.draw_detections(frame, results)
            
            # 计算FPS
            self.calculate_fps()
            
            # 显示结果
            cv2.imshow('YOLO实时目标检测', frame_with_detections)
            
            # 键盘输入处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = int(time.time())
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"已保存: {filename}")
        
        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = CameraDetector(camera_id=0)
    detector.run()
