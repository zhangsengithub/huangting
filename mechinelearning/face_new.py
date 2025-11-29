import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import cv2
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 人脸识别神经网络
class FaceRecognitionNN(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(FaceRecognitionNN, self).__init__()

        # 使用预训练的ResNet作为特征提取器
        self.backbone = models.resnet18(pretrained=True)

        # 修改最后的全连接层以适应人脸识别任务
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除原始分类层

        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def extract_features(self, x):
        """提取人脸特征向量"""
        with torch.no_grad():
            features = self.backbone(x)
        return features


# 修复后的人脸验证模块
class FaceValidator:
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()

    def predict_single_face(self, image_tensor):
        """预测单张人脸图像"""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_prob, predicted_class = torch.max(probabilities, 1)

            return predicted_class.item(), predicted_prob.item()

    def test_accuracy(self, test_loader):
        """测试模型准确率"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'人脸识别准确率: {accuracy:.2f}%')
        return accuracy

    def visualize_predictions(self, test_loader, num_samples=5):
        """可视化人脸识别结果 - 修复版本"""
        self.model.eval()
        data_iter = iter(test_loader)
        images, labels = next(data_iter)

        # 修复：确保不超出实际样本数量
        actual_samples = min(num_samples, images.size(0))
        if actual_samples == 0:
            print("没有可用的样本进行可视化")
            return

        fig, axes = plt.subplots(1, actual_samples, figsize=(20, 4))
        if actual_samples == 1:
            axes = [axes]

        with torch.no_grad():
            for i in range(actual_samples):
                image = images[i].to(self.device)
                true_label_idx = labels[i].item()

                # 预测
                predicted_idx, confidence = self.predict_single_face(image)

                # 获取类别名称
                true_label_name = self.class_names[true_label_idx]
                predicted_name = self.class_names[predicted_idx]

                # 显示图像
                ax = axes[i]
                # 反标准化显示图像
                img = image.cpu().squeeze().permute(1, 2, 0)
                img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                img = torch.clamp(img, 0, 1)

                ax.imshow(img)
                color = 'green' if predicted_idx == true_label_idx else 'red'
                ax.set_title(f'真实: {true_label_name}\n预测: {predicted_name}\n置信度: {confidence:.2f}',
                             color=color, fontsize=10)
                ax.axis('off')

        plt.tight_layout()
        plt.suptitle('人脸识别结果可视化', fontsize=14, y=1.02)
        plt.show()

    def simple_display_predictions(self, test_loader, num_samples=5):
        """简单的文本方式显示预测结果"""
        self.model.eval()
        data_iter = iter(test_loader)
        images, labels = next(data_iter)

        actual_samples = min(num_samples, images.size(0))

        print(f"\n=== 预测结果展示 (显示 {actual_samples} 个样本) ===")

        with torch.no_grad():
            for i in range(actual_samples):
                image = images[i].to(self.device)
                true_label_idx = labels[i].item()

                predicted_idx, confidence = self.predict_single_face(image)

                true_label_name = self.class_names[true_label_idx]
                predicted_name = self.class_names[predicted_idx]

                status = "✓ 正确" if predicted_idx == true_label_idx else "✗ 错误"
                print(f"样本 {i + 1}:")
                print(f"  真实: {true_label_name}")
                print(f"  预测: {predicted_name}")
                print(f"  置信度: {confidence:.4f}")
                print(f"  结果: {status}")
                print("-" * 40)


# 中文人脸数据集 - 只使用中文目录
class ChineseFaceDataset(Dataset):
    def __init__(self, root_dir='./data/faces', transform=None, max_samples_per_class=50):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # 预定义的中文人物名称
        self.chinese_names = ['张三', '李四', '王五', '赵六', '孙七']
        
        # 只从中文目录加载图像
        self.load_chinese_faces_only(max_samples_per_class)

    def load_chinese_faces_only(self, max_samples_per_class):
        """只从中文目录加载人脸图像"""
        if not os.path.exists(self.root_dir):
            print(f"数据目录不存在: {self.root_dir}")
            return

        # 只加载预定义的中文目录
        valid_classes = []
        for chinese_name in self.chinese_names:
            chinese_dir = os.path.join(self.root_dir, chinese_name)
            if os.path.exists(chinese_dir):
                valid_classes.append(chinese_name)
                print(f"找到中文目录: {chinese_name}")
            else:
                print(f"警告: 中文目录不存在: {chinese_name}")

        if len(valid_classes) == 0:
            print("未找到任何中文人物目录")
            return

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        print(f"\n正在从 {len(valid_classes)} 个中文目录加载图像:")
        total_images = 0
        for cls_name in valid_classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # 限制每个类别的样本数量
            image_files = image_files[:max_samples_per_class]

            print(f"  {cls_name}: {len(image_files)} 张图像")

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                self.data.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])
                total_images += 1

        print(f"总共加载 {len(self.data)} 张中文人物人脸图像")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            # 如果加载失败，返回一个随机图像
            print(f"加载图像失败 {img_path}: {e}")
            dummy_image = torch.randn(3, 224, 224)
            return dummy_image, label


# 使用网络摄像头采集人脸数据
class FaceDataCollector:
    def __init__(self, output_dir='./collected_faces'):
        self.output_dir = output_dir
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        os.makedirs(output_dir, exist_ok=True)

    def collect_faces(self, person_name, num_faces=50):
        """使用摄像头采集人脸数据"""
        person_dir = os.path.join(self.output_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        face_count = 0

        print(f"开始采集 {person_name} 的人脸数据...")
        print("请面对摄像头，按 'c' 采集，按 'q' 退出")

        while face_count < num_faces:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 显示计数
                cv2.putText(frame, f'Faces: {face_count}/{num_faces}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Face Collection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(faces) > 0:
                # 保存人脸图像
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y + h, x:x + w]
                    # 调整大小为标准尺寸
                    face_resized = cv2.resize(face_roi, (224, 224))
                    face_filename = os.path.join(person_dir, f'face_{face_count:04d}.jpg')
                    print(face_filename)
                    cv2.imwrite(face_filename, face_resized)
                    face_count += 1
                    print(f"已采集 {face_count}/{num_faces} 张人脸")
                    break

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"完成采集 {face_count} 张 {person_name} 的人脸图像")


# 主程序
def main():
    # 超参数设置
    batch_size = 8  # 使用较小的批次大小
    learning_rate = 0.001
    num_epochs = 10
    feature_dim = 512

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("=" * 60)
    print("          中文人脸识别系统")
    print("=" * 60)
    
    # 显示将要使用的中文人物
    chinese_names = ['张三', '李四', '王五', '赵六', '孙七']
    print(f"\n系统将使用以下 {len(chinese_names)} 个中文人物:")
    for i, person in enumerate(chinese_names, 1):
        print(f"  {i}. {person}")

    # 加载中文人脸数据集
    print(f"\n正在加载中文人脸数据集...")
    dataset = ChineseFaceDataset(transform=transform, max_samples_per_class=100)

    if len(dataset) == 0:
        print("未找到中文人脸数据")
        
        # 提供数据采集选项
        collector = FaceDataCollector()
        response = input("是否使用摄像头采集人脸数据？(y/n): ")
        if response.lower() == 'y':
            person_name = input("请输入中文人物姓名: ")
            collector.collect_faces(person_name, num_faces=50)
            print("请重新运行程序以使用采集的数据进行训练")
            return
        else:
            print("程序退出")
            return

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # 确保测试集至少有批次大小那么多的样本
    if test_size < batch_size:
        batch_size = max(1, test_size)
        print(f"调整批次大小为: {batch_size}")

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 获取类别名称
    class_names = list(dataset.idx_to_class.values())
    num_classes = len(class_names)

    print(f"\n数据集信息:")
    print(f"  中文人物数量: {num_classes}")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  测试样本: {len(test_dataset)}")
    print(f"  批次大小: {batch_size}")
    print(f"  人物列表: {class_names}")

    # 初始化设备、模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    model = FaceRecognitionNN(num_classes=num_classes, feature_dim=feature_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练循环
    print(f"\n开始训练中文人脸识别模型...")
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                acc = 100 * correct / total
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], '
                      f'Loss: {loss.item():.4f}, Acc: {acc:.2f}%')

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}] 平均损失: {avg_loss:.4f}, 准确率: {epoch_acc:.2f}%')

    print("训练完成!")

    # 创建验证器并测试模型
    validator = FaceValidator(model, device, class_names)

    # 1. 测试准确率
    print("\n" + "=" * 50)
    print("模型性能评估")
    print("=" * 50)
    validator.test_accuracy(test_loader)

    # 2. 可视化预测结果
    print("\n" + "=" * 50)
    print("中文人脸识别结果可视化")
    print("=" * 50)
    validator.visualize_predictions(test_loader, num_samples=min(5, batch_size))

    # 3. 文本方式显示结果
    validator.simple_display_predictions(test_loader, num_samples=min(5, batch_size))

    # 4. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_names': class_names,
        'feature_dim': feature_dim
    }, 'chinese_face_recognition_model.pth')
    print(f"\n模型已保存为 'chinese_face_recognition_model.pth'")

    print(f"\n训练完成！模型可以识别 {num_classes} 个中文人物: {class_names}")


if __name__ == "__main__":
    main()