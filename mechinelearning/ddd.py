import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
from matplotlib.font_manager import FontProperties

# 尝试多种方法设置中文字体
def setup_chinese_font():
    """设置中文字体，解决乱码问题"""
    try:
        # 方法1: 直接使用系统字体路径
        font_paths = [
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Ubuntu
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',  # Ubuntu
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Ubuntu 文泉驿
            '/System/Library/Fonts/Arial.ttf',  # macOS
            'C:/Windows/Fonts/simhei.ttf',  # Windows
            'C:/Windows/Fonts/msyh.ttc',  # Windows
        ]
        
        # 方法2: 尝试找到可用的中文字体
        available_fonts = []
        for font_path in font_paths:
            if os.path.exists(font_path):
                available_fonts.append(font_path)
        
        if available_fonts:
            # 使用第一个可用的字体
            chinese_font = FontProperties(fname=available_fonts[0])
            plt.rcParams['font.family'] = [chinese_font.get_name()]
            print(f"✅ 使用字体: {available_fonts[0]}")
            return available_fonts[0]
        else:
            # 方法3: 使用matplotlib内置字体
            plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'sans-serif']
            return None
        
        plt.rcParams['axes.unicode_minus'] = False
        
    except Exception as e:
        print(f"⚠️ 字体设置警告: {e}")
        print("使用默认字体，中文可能显示为方块")
        return None

# 调用字体设置
font_path = setup_chinese_font()

# 定义与训练时相同的模型结构
class FaceRecognitionNN(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(FaceRecognitionNN, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def safe_chinese_display(text):
    """安全显示中文文本，如果显示失败则使用英文替代"""
    try:
        return text
    except:
        # 如果中文显示失败，返回英文替代
        chinese_to_english = {
            "模型内容分析": "Model Analysis",
            "基本信息": "Basic Info",
            "类别数量": "Number of Classes",
            "人物列表": "Class Names",
            "特征维度": "Feature Dimension",
            "模型参数统计": "Model Parameters",
            "参数层数": "Parameter Layers",
            "参数详情": "Parameter Details",
            "总参数量": "Total Parameters",
            "模型结构可视化": "Model Structure Visualization",
            "模型结构": "Model Structure",
            "权重分布": "Weight Distribution",
            "偏置分布": "Bias Distribution",
            "各层参数数量": "Parameters per Layer",
            "权重值": "Weight Values",
            "频次": "Frequency",
            "偏置值": "Bias Values",
            "层索引": "Layer Index",
            "参数数量": "Parameter Count",
            "图片预测测试": "Image Prediction Test",
            "测试数据目录不存在": "Test directory not found",
            "未找到测试图片": "No test images found",
            "创建随机测试数据": "Creating random test data",
            "找到测试图片": "Found test images",
            "处理图片时出错": "Error processing image",
            "测试准确率": "Test Accuracy",
            "使用随机数据进行测试": "Testing with random data",
            "随机输入预测结果": "Random Input Prediction",
            "预测人物": "Predicted Person",
            "置信度": "Confidence",
            "所有类别概率": "All Class Probabilities",
            "实时摄像头测试": "Real-time Camera Test",
            "是否启动摄像头测试": "Start camera test?",
            "摄像头已启动": "Camera started",
            "按退出": "Press to quit",
            "预测错误": "Prediction error",
            "开始测试模型": "Start testing model",
            "模型加载成功": "Model loaded successfully",
            "加载模型失败": "Failed to load model",
            "模型测试完成": "Model testing completed"
        }
        return chinese_to_english.get(text, text)

def cv2_add_chinese_text(img, text, position, font_size=30, color=(0, 255, 0)):
    """
    在OpenCV图像上添加中文文本
    使用PIL绘制中文，然后转换回OpenCV格式
    """
    try:
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试加载中文字体
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 使用默认字体（可能不支持中文）
            font = ImageFont.load_default()
            # 如果默认字体不支持中文，使用简单的英文替代
            text = safe_chinese_display(text)
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color)
        
        # 转换回OpenCV格式
        img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_with_text
    except Exception as e:
        # 如果出错，使用英文替代
        print(f"⚠️ 中文文本绘制失败，使用英文: {e}")
        english_text = safe_chinese_display(text)
        cv2.putText(img, english_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

def load_and_analyze_model():
    """加载并分析模型内容"""
    print("=" * 60)
    print(safe_chinese_display("          模型内容分析"))
    print("=" * 60)
    
    try:
        # 加载模型
        checkpoint = torch.load('chinese_face_model.pth', map_location='cpu')
        
        print("✅ " + safe_chinese_display("模型加载成功！"))
        print("\n=== " + safe_chinese_display("基本信息") + " ===")
        print(f"📊 " + safe_chinese_display("类别数量") + f": {checkpoint['num_classes']}")
        
        # 安全显示人物列表
        class_names = checkpoint['class_names']
        print(f"👥 " + safe_chinese_display("人物列表") + f": ", end="")
        try:
            print(class_names)
        except:
            print("[中文名称 - 需要正确字体支持]")
        
        if 'feature_dim' in checkpoint:
            print(f"🔢 " + safe_chinese_display("特征维度") + f": {checkpoint['feature_dim']}")
        
        # 模型参数统计
        model_params = checkpoint['model_state_dict']
        print(f"\n=== " + safe_chinese_display("模型参数统计") + " ===")
        print(f"📁 " + safe_chinese_display("参数层数") + f": {len(model_params)}")
        
        total_params = 0
        print(f"\n📋 " + safe_chinese_display("参数详情") + ":")
        for name, param in model_params.items():
            param_count = param.numel()
            total_params += param_count
            print(f"   {name:<50} {str(param.shape):<20} {param_count:>8,} " + safe_chinese_display("参数"))
        
        print(f"\n💾 " + safe_chinese_display("总参数量") + f": {total_params:,}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ " + safe_chinese_display("加载模型失败") + f": {e}")
        return None

def visualize_model_structure(checkpoint):
    """可视化模型结构"""
    print("\n" + "=" * 50)
    print(safe_chinese_display("模型结构可视化"))
    print("=" * 50)
    
    # 创建模型实例
    model = FaceRecognitionNN(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 打印模型结构
    print("🧩 " + safe_chinese_display("模型结构") + ":")
    print(model)
    
    # 显示参数分布
    weights = []
    biases = []
    layers = []
    
    for name, param in checkpoint['model_state_dict'].items():
        if 'weight' in name:
            weights.extend(param.flatten().tolist())
            layers.append(name)
        elif 'bias' in name:
            biases.extend(param.flatten().tolist())
    
    # 绘制权重分布
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(weights, bins=50, alpha=0.7, color='blue')
    plt.title(safe_chinese_display('权重分布'))
    plt.xlabel(safe_chinese_display('权重值'))
    plt.ylabel(safe_chinese_display('频次'))
    
    plt.subplot(1, 3, 2)
    plt.hist(biases, bins=50, alpha=0.7, color='green')
    plt.title(safe_chinese_display('偏置分布'))
    plt.xlabel(safe_chinese_display('偏置值'))
    plt.ylabel(safe_chinese_display('频次'))
    
    plt.subplot(1, 3, 3)
    layer_sizes = [param.numel() for name, param in checkpoint['model_state_dict'].items() if 'weight' in name]
    plt.bar(range(len(layer_sizes)), layer_sizes, color='orange', alpha=0.7)
    plt.title(safe_chinese_display('各层参数数量'))
    plt.xlabel(safe_chinese_display('层索引'))
    plt.ylabel(safe_chinese_display('参数数量'))
    plt.xticks(range(len(layer_sizes)), [f'L{i+1}' for i in range(len(layer_sizes))], rotation=45)
    
    plt.tight_layout()
    
    # 尝试保存图片，避免显示问题
    try:
        plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
        print("💾 模型分析图已保存为 'model_analysis.png'")
    except:
        pass
    
    plt.show()

def test_with_sample_images(checkpoint):
    """使用测试图片进行预测"""
    print("\n" + "=" * 50)
    print(safe_chinese_display("图片预测测试"))
    print("=" * 50)
    
    # 创建模型实例
    model = FaceRecognitionNN(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    class_names = checkpoint['class_names']
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 测试数据目录
    test_dir = './data/faces'
    
    if not os.path.exists(test_dir):
        print("❌ " + safe_chinese_display("测试数据目录不存在"))
        # 创建一些随机测试数据
        print("🎨 " + safe_chinese_display("创建随机测试数据") + "...")
        test_random_predictions(model, class_names, transform)
        return
    
    # 收集测试图片
    test_images = []
    for person in class_names:
        person_dir = os.path.join(test_dir, person)
        if os.path.exists(person_dir):
            image_files = [f for f in os.listdir(person_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in image_files[:2]:  # 每个类别取2张
                test_images.append((os.path.join(person_dir, img_file), person))
    
    if not test_images:
        print("❌ " + safe_chinese_display("未找到测试图片"))
        # 创建一些随机测试数据
        print("🎨 " + safe_chinese_display("创建随机测试数据") + "...")
        test_random_predictions(model, class_names, transform)
        return
    
    print(f"📸 " + safe_chinese_display("找到测试图片") + f" {len(test_images)} 张")
    
    # 进行预测
    correct = 0
    total = 0
    
    plt.figure(figsize=(15, 10))
    
    for idx, (img_path, true_label) in enumerate(test_images[:9]):  # 最多显示9张
        try:
            # 加载和预处理图片
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_prob, predicted_class = torch.max(probabilities, 1)
                
                predicted_label = class_names[predicted_class.item()]
                confidence = predicted_prob.item()
            
            # 显示结果
            plt.subplot(3, 3, idx + 1)
            
            # 反标准化显示图像
            img_display = input_tensor.squeeze().permute(1, 2, 0)
            img_display = img_display * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img_display = torch.clamp(img_display, 0, 1)
            
            plt.imshow(img_display)
            
            # 设置标题颜色
            color = 'green' if predicted_label == true_label else 'red'
            status = "✓" if predicted_label == true_label else "✗"
            
            # 安全显示中文标签
            try:
                title = f'{status} 真实: {true_label}\n预测: {predicted_label}\n置信度: {confidence:.3f}'
                plt.title(title, color=color, fontsize=10)
            except:
                # 如果中文显示失败，使用简单标签
                title = f'{status} True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.3f}'
                plt.title(title, color=color, fontsize=8)
            
            plt.axis('off')
            
            total += 1
            if predicted_label == true_label:
                correct += 1
                
        except Exception as e:
            print(f"❌ " + safe_chinese_display("处理图片时出错") + f" {img_path}: {e}")
    
    plt.tight_layout()
    
    # 安全设置总标题
    try:
        plt.suptitle(safe_chinese_display('人脸识别测试结果'), fontsize=16, y=1.02)
    except:
        plt.suptitle('Face Recognition Test Results', fontsize=16, y=1.02)
    
    # 尝试保存结果图片
    try:
        plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
        print("💾 测试结果图已保存为 'test_results.png'")
    except:
        pass
    
    plt.show()
    
    if total > 0:
        accuracy = 100 * correct / total
        print(f"\n🎯 " + safe_chinese_display("测试准确率") + f": {accuracy:.1f}% ({correct}/{total})")

def test_random_predictions(model, class_names, transform):
    """使用随机数据进行预测测试"""
    print("\n🔧 " + safe_chinese_display("使用随机数据进行测试") + "...")
    
    # 创建随机输入
    random_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(random_input)
        probabilities = torch.softmax(output, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, 1)
        
        predicted_label = class_names[predicted_class.item()]
        confidence = predicted_prob.item()
    
    print(f"🎲 " + safe_chinese_display("随机输入预测结果") + ":")
    print(f"   " + safe_chinese_display("预测人物") + f": {predicted_label}")
    print(f"   " + safe_chinese_display("置信度") + f": {confidence:.4f}")
    
    # 显示所有类别的概率
    print(f"\n📊 " + safe_chinese_display("所有类别概率") + ":")
    probs = probabilities.squeeze().tolist()
    for i, (cls_name, prob) in enumerate(zip(class_names, probs)):
        print(f"   {cls_name}: {prob:.4f}")

def real_time_camera_test(checkpoint):
    """实时摄像头测试"""
    print("\n" + "=" * 50)
    print(safe_chinese_display("实时摄像头测试"))
    print("=" * 50)
    
    response = input(safe_chinese_display("是否启动摄像头测试？") + "(y/n): ")
    if response.lower() != 'y':
        return
    
    # 创建模型实例
    model = FaceRecognitionNN(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    class_names = checkpoint['class_names']
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("📹 " + safe_chinese_display("摄像头已启动") + ", " + safe_chinese_display("按") + " 'q' " + safe_chinese_display("退出") + "...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 创建一个用于显示的帧副本
        display_frame = frame.copy()
        
        # 人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # 预处理
                face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0)
                
                # 预测
                with torch.no_grad():
                    output = model(face_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_prob, predicted_class = torch.max(probabilities, 1)
                    
                    predicted_label = class_names[predicted_class.item()]
                    confidence = predicted_prob.item()
                
                # 绘制矩形
                color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # 使用中文文本绘制函数
                label_text = f'{predicted_label}: {confidence:.2f}'
                display_frame = cv2_add_chinese_text(
                    display_frame, 
                    label_text, 
                    (x, y-30), 
                    font_size=20, 
                    color=color
                )
                
            except Exception as e:
                print(f"❌ " + safe_chinese_display("预测错误") + f": {e}")
        
        # 显示帧
        cv2.imshow('Face Recognition Test', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 主测试函数
def main():
    print("🚀 " + safe_chinese_display("开始测试模型") + " chinese_face_model.pth...")
    
    # 1. 加载和分析模型
    checkpoint = load_and_analyze_model()
    if checkpoint is None:
        return
    
    # 2. 可视化模型结构
    visualize_model_structure(checkpoint)
    
    # 3. 使用测试图片进行预测
    test_with_sample_images(checkpoint)
    
    # 4. 实时摄像头测试（可选）
    real_time_camera_test(checkpoint)
    
    print("\n" + "=" * 60)
    print("✅ " + safe_chinese_display("模型测试完成！"))
    print("=" * 60)

if __name__ == "__main__":
    main()