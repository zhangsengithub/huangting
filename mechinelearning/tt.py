import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def load_and_analyze_model():
    """加载并分析模型内容"""
    print("=" * 60)
    print("          模型内容分析")
    print("=" * 60)
    
    try:
        # 加载模型
        checkpoint = torch.load('chinese_face_model.pth', map_location='cpu')
        
        print("✅ 模型加载成功！")
        print("\n=== 基本信息 ===")
        print(f"📊 类别数量: {checkpoint['num_classes']}")
        print(f"👥 人物列表: {checkpoint['class_names']}")
        
        if 'feature_dim' in checkpoint:
            print(f"🔢 特征维度: {checkpoint['feature_dim']}")
        
        # 模型参数统计
        model_params = checkpoint['model_state_dict']
        print(f"\n=== 模型参数统计 ===")
        print(f"📁 参数层数: {len(model_params)}")
        
        total_params = 0
        print(f"\n📋 参数详情:")
        for name, param in model_params.items():
            param_count = param.numel()
            total_params += param_count
            print(f"   {name:<50} {str(param.shape):<20} {param_count:>8,} 参数")
        
        print(f"\n💾 总参数量: {total_params:,}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

def visualize_model_structure(checkpoint):
    """可视化模型结构"""
    print("\n" + "=" * 50)
    print("模型结构可视化")
    print("=" * 50)
    
    # 创建模型实例
    model = FaceRecognitionNN(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 打印模型结构
    print("🧩 模型结构:")
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
    plt.title('权重分布')
    plt.xlabel('权重值')
    plt.ylabel('频次')
    
    plt.subplot(1, 3, 2)
    plt.hist(biases, bins=50, alpha=0.7, color='green')
    plt.title('偏置分布')
    plt.xlabel('偏置值')
    plt.ylabel('频次')
    
    plt.subplot(1, 3, 3)
    layer_sizes = [param.numel() for name, param in checkpoint['model_state_dict'].items() if 'weight' in name]
    plt.bar(range(len(layer_sizes)), layer_sizes, color='orange', alpha=0.7)
    plt.title('各层参数数量')
    plt.xlabel('层索引')
    plt.ylabel('参数数量')
    plt.xticks(range(len(layer_sizes)), [f'L{i+1}' for i in range(len(layer_sizes))], rotation=45)
    
    plt.tight_layout()
    plt.show()

def test_with_sample_images(checkpoint):
    """使用测试图片进行预测"""
    print("\n" + "=" * 50)
    print("图片预测测试")
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
        print("❌ 测试数据目录不存在")
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
        print("❌ 未找到测试图片")
        # 创建一些随机测试数据
        print("🎨 创建随机测试数据...")
        test_random_predictions(model, class_names, transform)
        return
    
    print(f"📸 找到 {len(test_images)} 张测试图片")
    
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
            
            plt.title(f'{status} 真实: {true_label}\n预测: {predicted_label}\n置信度: {confidence:.3f}', 
                     color=color, fontsize=10)
            plt.axis('off')
            
            total += 1
            if predicted_label == true_label:
                correct += 1
                
        except Exception as e:
            print(f"❌ 处理图片 {img_path} 时出错: {e}")
    
    plt.tight_layout()
    plt.suptitle('人脸识别测试结果', fontsize=16, y=1.02)
    plt.show()
    
    if total > 0:
        accuracy = 100 * correct / total
        print(f"\n🎯 测试准确率: {accuracy:.1f}% ({correct}/{total})")

def test_random_predictions(model, class_names, transform):
    """使用随机数据进行预测测试"""
    print("\n🔧 使用随机数据进行测试...")
    
    # 创建随机输入
    random_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(random_input)
        probabilities = torch.softmax(output, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, 1)
        
        predicted_label = class_names[predicted_class.item()]
        confidence = predicted_prob.item()
    
    print(f"🎲 随机输入预测结果:")
    print(f"   预测人物: {predicted_label}")
    print(f"   置信度: {confidence:.4f}")
    
    # 显示所有类别的概率
    print(f"\n📊 所有类别概率:")
    probs = probabilities.squeeze().tolist()
    for i, (cls_name, prob) in enumerate(zip(class_names, probs)):
        print(f"   {cls_name}: {prob:.4f}")

def real_time_camera_test(checkpoint):
    """实时摄像头测试"""
    print("\n" + "=" * 50)
    print("实时摄像头测试")
    print("=" * 50)
    
    response = input("是否启动摄像头测试？(y/n): ")
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
    print("📹 摄像头已启动，按 'q' 退出...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
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
                
                # 绘制结果
                color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f'{predicted_label}: {confidence:.2f}', 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            except Exception as e:
                print(f"预测错误: {e}")
        
        cv2.imshow('Face Recognition Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 主测试函数
def main():
    print("🚀 开始测试 chinese_face_model.pth 模型...")
    
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
    print("✅ 模型测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
