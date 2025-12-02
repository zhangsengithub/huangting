from ultralytics import YOLO
import cv2
import time
import numpy as np
import threading
import queue
from PIL import Image, ImageDraw, ImageFont
import platform
import torch


class StableYOLODisplay:
    def __init__(self):
        self.model = YOLO("yolo12n.pt")
        self.cap = cv2.VideoCapture(0)

        # 降低摄像头分辨率减少处理负担
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

        # 检测GPU可用性
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "未知GPU"
            print(f"✅ GPU检测到: {gpu_name}")
            self.device = "cuda"
        else:
            print("⚠️  未检测到GPU，将使用CPU进行推理")
            self.device = "cpu"

        # 用于存储最新检测结果
        self.latest_detections = []
        self.last_detection_time = 0
        self.detection_lock = threading.Lock()

        # 显示相关
        self.window_name = "YOLO12稳定检测"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        # 加载中文字体
        self.font = self._load_chinese_font()

        # 性能统计
        self.frame_count = 0
        self.fps = 0
        self.detection_fps = 0
        self.start_time = time.time()

        # 用于多线程的队列
        self.frame_queue = queue.Queue(maxsize=3)  # 增加队列大小以适应GPU处理
        self.running = True

        print("✅ YOLO模型加载成功")
        print(f"📷 摄像头已打开 (1024x768)")
        print(f"🔄 启动稳定检测模式，使用设备: {self.device.upper()}")
        print(f"🎯 当前系统: {platform.system()}")

    def _load_chinese_font(self):
        """加载中文字体"""
        system_name = platform.system()

        # 尝试不同的字体路径
        font_paths = [
            "msyh.ttc",  # 微软雅黑
            "simhei.ttf",  # 黑体
            "simsun.ttc",  # 宋体
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        ]

        # 根据系统添加特定字体路径
        if system_name == "Windows":
            font_paths.insert(0, "C:/Windows/Fonts/msyh.ttc")
            font_paths.insert(1, "C:/Windows/Fonts/simhei.ttf")
        elif system_name == "Darwin":  # macOS
            font_paths.insert(0, "/System/Library/Fonts/PingFang.ttc")
            font_paths.insert(1, "/Library/Fonts/Arial Unicode.ttf")

        font_size = 20

        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                print(f"✅ 加载字体成功: {font_path}")
                return font
            except Exception as e:
                continue

        # 如果没有找到字体，使用默认字体
        print("⚠️  未找到中文字体，使用默认字体")
        return ImageFont.load_default()

    def put_chinese_text(self, image, text, position, font_color=(255, 255, 255)):
        """在图像上绘制中文文本"""
        # 将OpenCV图像转换为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 绘制文本
        draw.text(position, text, font=self.font, fill=font_color)

        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def detection_thread(self):
        """独立的检测线程，使用GPU加速"""
        detection_count = 0
        detection_start_time = time.time()

        while self.running:
            try:
                # 从队列获取帧
                frame = self.frame_queue.get(timeout=0.5)

                # 使用GPU进行高分辨率推理
                results = self.model(frame, verbose=False,
                                     imgsz=640,  # 高分辨率
                                     conf=0.5,  # 置信度阈值
                                     device=self.device,
                                     half=False)  # 使用半精度可进一步提高速度，但可能需要更多显存

                detection_count += 1

                # 计算检测FPS
                if detection_count % 10 == 0:
                    self.detection_fps = 10 / (time.time() - detection_start_time)
                    detection_start_time = time.time()

                with self.detection_lock:
                    if results and len(results) > 0 and results[0].boxes is not None:
                        # 提取检测信息并转换为可序列化格式
                        self.latest_detections = []
                        for box in results[0].boxes:
                            try:
                                # 转换为Python原生类型
                                xyxy = box.xyxy[0].cpu().numpy().tolist()
                                conf = box.conf[0].item()
                                cls_id = box.cls[0].item()

                                self.latest_detections.append({
                                    'xyxy': xyxy,
                                    'conf': conf,
                                    'cls': cls_id
                                })
                            except:
                                continue
                    else:
                        self.latest_detections = []
                    self.last_detection_time = time.time()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"检测线程错误: {e}")
                import traceback
                traceback.print_exc()

    def run(self):
        # 启动检测线程
        detect_thread = threading.Thread(target=self.detection_thread, daemon=True)
        detect_thread.start()

        print("\n🎯 稳定检测已启动")
        print("   - 按 'q' 键退出")
        print("   - 按 's' 键保存当前帧")
        print("   - 按 'd' 键切换显示模式")
        print("   - 按 'r' 键切换分辨率模式")
        print("   - 按 'g' 键切换设备 (GPU/CPU)")

        display_mode = 1  # 0:原始画面, 1:带检测框, 2:仅检测结果
        resolution_mode = 0  # 0:640x640, 1:800x800, 2:1024x1024

        resolution_modes = [
            (640, "640x640 (平衡)"),
            (800, "800x800 (高精度)"),
            (1024, "1024x1024 (最高精度)"),
        ]

        last_fps_update = time.time()
        fps_frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break

            self.frame_count += 1
            fps_frame_count += 1

            # 发送帧到检测队列（非阻塞方式）
            try:
                if self.frame_queue.qsize() < 3:
                    self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # 队列已满，跳过这帧

            # 计算FPS（每秒更新一次）
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                self.fps = fps_frame_count / (current_time - last_fps_update)
                fps_frame_count = 0
                last_fps_update = current_time

            # 创建显示帧
            display_frame = frame.copy()

            # 获取最新检测结果
            with self.detection_lock:
                detections = list(self.latest_detections)  # 转换为新列表
                detection_age = current_time - self.last_detection_time

            # 绘制检测框（使用最近的结果）
            if detections and display_mode > 0:
                for detection in detections[:10]:  # 最多显示10个检测
                    try:
                        x1, y1, x2, y2 = map(int, detection['xyxy'])
                        conf = detection['conf']
                        cls_id = int(detection['cls'])

                        # 根据置信度设置颜色
                        color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)  # 绿色或橙色

                        # 绘制检测框
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                        # 标签背景
                        label = f"{self.model.names[cls_id]} {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        cv2.rectangle(display_frame,
                                      (x1, y1 - text_height - 10),
                                      (x1 + text_width + 10, y1),
                                      color, -1)

                        cv2.putText(display_frame, label,
                                    (x1 + 5, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    except Exception as e:
                        continue

            # 创建状态栏
            status_bar_height = 120
            status_bar = np.zeros((status_bar_height, display_frame.shape[1], 3), dtype=np.uint8)

            # 使用PIL绘制中文文本到状态栏
            device_name = "GPU" if self.gpu_available else "CPU"
            status_bar = self.put_chinese_text(status_bar, f"设备: {device_name}", (20, 20))

            status_bar = self.put_chinese_text(status_bar, f"显示帧率: {self.fps:.1f} FPS", (20, 50))
            status_bar = self.put_chinese_text(status_bar, f"检测帧率: {self.detection_fps:.1f} FPS", (20, 80))

            detection_info = f"检测目标: {len(detections)}个 | 延迟: {detection_age:.2f}秒"
            status_bar = self.put_chinese_text(status_bar, detection_info, (250, 50))

            mode_texts = ["原始画面", "检测模式", "专注模式"]
            status_bar = self.put_chinese_text(status_bar, f"显示模式: {mode_texts[display_mode]}", (250, 80))

            # 显示当前分辨率模式
            res_mode_name = resolution_modes[resolution_mode][1]
            status_bar = self.put_chinese_text(status_bar, f"分辨率: {res_mode_name}", (500, 50))

            # 操作提示
            help_text = "Q:退出  S:保存截图  D:切换显示模式  R:切换分辨率  G:切换设备"
            status_bar = self.put_chinese_text(status_bar, help_text, (500, 80), font_color=(200, 200, 200))

            # 合并状态栏和画面
            display_frame = cv2.vconcat([status_bar, display_frame])

            # 显示画面
            cv2.imshow(self.window_name, display_frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 用户退出程序")
                break
            elif key == ord('s'):
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 已保存截图: {filename}")
            elif key == ord('d'):
                display_mode = (display_mode + 1) % 3
                mode_names = ["原始画面", "检测画面", "专注模式"]
                print(f"🔄 切换到显示模式: {mode_names[display_mode]}")
            elif key == ord('r'):
                resolution_mode = (resolution_mode + 1) % len(resolution_modes)
                res_size, res_name = resolution_modes[resolution_mode]

                # 更新模型推理分辨率
                def update_model_resolution():
                    try:
                        # 重新运行检测线程以应用新的分辨率
                        self.model.imgsz = res_size
                        print(f"🔄 切换分辨率模式到: {res_name}")
                        print(f"   推理分辨率: {res_size}")
                    except Exception as e:
                        print(f"切换分辨率失败: {e}")

                # 启动一个线程来更新分辨率，避免阻塞主线程
                update_thread = threading.Thread(target=update_model_resolution, daemon=True)
                update_thread.start()
            elif key == ord('g'):
                # 切换设备 (GPU/CPU)
                if self.gpu_available:
                    if self.device == "cuda":
                        self.device = "cpu"
                        print(f"🔄 切换到CPU进行推理")
                    else:
                        self.device = "cuda"
                        print(f"🔄 切换到GPU进行推理")
                else:
                    print("⚠️  没有可用的GPU设备")

        # 清理资源
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

        # 显示统计数据
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time
        print(f"\n📊 运行统计:")
        print(f"   总帧数: {self.frame_count}")
        print(f"   运行时间: {total_time:.1f}秒")
        print(f"   平均FPS: {avg_fps:.1f}")
        print(f"   最终使用设备: {self.device.upper()}")


def main():
    print("=" * 50)
    print("YOLO12 稳定摄像头检测系统")
    print("=" * 50)

    # 测试摄像头
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("❌ 无法打开摄像头")
        print("请检查:")
        print("  1. 摄像头是否连接")
        print("  2. 是否有其他程序占用摄像头")
        print("  3. 尝试重启电脑")
        return
    test_cap.release()

    # 运行主程序
    try:
        yolo_display = StableYOLODisplay()
        yolo_display.run()
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序已结束")


if __name__ == "__main__":
    main()