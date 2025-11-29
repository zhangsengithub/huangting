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


# 安全的人脸识别神经网络（修复BatchNorm问题）
class SafeFaceRecognitionNN(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(SafeFaceRecognitionNN, self).__init__()

        # 使用预训练的ResNet作为特征提取器
        self.backbone = models.resnet18(pretrained=True)

        # 修改最后的全连接层以适应人脸识别任务
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 使用安全的归一化方法
        self.classifier = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.LayerNorm(feature_dim),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


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
                # 确保批次大小足够
                if images.size(0) == 1:
                    continue  # 跳过单个样本的批次

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if total == 0:
            print("没有有效的测试样本")
            return 0

        accuracy = 100 * correct / total
        print(f'人脸识别准确率: {accuracy:.2f}%')
        return accuracy


# 中文人脸数据集
class ChineseFaceDataset(Dataset):
    def __init__(self, root_dir='/home/zhangsen/图片/img_align_celeba', transform=None, max_samples_per_class=50):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.chinese_names = ['张森', '张展']
        self.load_chinese_faces_only(max_samples_per_class)

    def load_chinese_faces_only_t(self, max_samples_per_class):
        """只从中文目录加载人脸图像"""
        if not os.path.exists(self.root_dir):
            print(f"数据目录不存在: {self.root_dir}")
            return

        valid_classes = []
        for chinese_name in self.chinese_names:
            chinese_dir = os.path.join(self.root_dir, chinese_name)
            if os.path.exists(chinese_dir):
                valid_classes.append(chinese_name)
                print(f"找到中文目录: {chinese_name}")

        if len(valid_classes) == 0:
            print("未找到任何中文人物目录")
            return

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        print(f"\n正在从 {len(valid_classes)} 个中文目录加载图像:")
        for cls_name in valid_classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files = image_files[:max_samples_per_class]

            print(f"  {cls_name}: {len(image_files)} 张图像")

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                self.data.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

        print(f"总共加载 {len(self.data)} 张中文人物人脸图像")

    def load_chinese_faces_only(self, max_samples_per_class):
        """只从中文目录加载人脸图像，使用文件名作为标签"""
        if not os.path.exists(self.root_dir):
            print(f"数据目录不存在: {self.root_dir}")
            return

        # 获取所有图片文件
        image_files = [f for f in os.listdir(self.root_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(image_files) == 0:
            print("未找到任何图像文件")
            return

        size = len(image_files);
        # 限制每个类别的样本数量（这里每个文件都是一个独立的"类别"）
        image_files = image_files[:size]

        # 创建标签映射：使用文件名（不带扩展名）作为标签
        self.class_to_idx = {os.path.splitext(f)[0]: i for i, f in enumerate(image_files)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        print(f"\n正在加载 {len(image_files)} 张人脸图像:")

        for img_file in image_files:
            img_path = os.path.join(self.root_dir, img_file)
            # 使用文件名（不带扩展名）作为标签
            label = os.path.splitext(img_file)[0]

            self.data.append(img_path)
            self.labels.append(self.class_to_idx[label])

        print(f"总共加载 {len(self.data)} 张人脸图像")
        print(f"类别数量: {len(self.class_to_idx)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            dummy_image = torch.randn(3, 224, 224)
            return dummy_image, label

# 主程序
def main():

    # 超参数设置 - 确保批次大小足够
    batch_size = 256  # 至少为2
    learning_rate = 0.001
    num_epochs = 15
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
    print("          中文人脸识别系统（修复版）")
    print("=" * 60)

    # 加载中文人脸数据集
    print(f"\n正在加载中文人脸数据集...")
    dataset = ChineseFaceDataset(transform=transform, max_samples_per_class=100)

    if len(dataset) == 0:
        print("未找到中文人脸数据")
        return

    # 确保有足够的数据
    if len(dataset) < batch_size:
        batch_size = max(2, len(dataset) // 2)
        print(f"数据量较少，调整批次大小为: {batch_size}")

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    if test_size < batch_size:
        batch_size = max(2, test_size)
        print(f"测试集较小，调整批次大小为: {batch_size}")

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

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

    # 使用修复后的模型
    model = SafeFaceRecognitionNN(num_classes=num_classes, feature_dim=feature_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 训练循环
    print(f"\n开始训练中文人脸识别模型...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            # 确保批次大小有效
            if images.size(0) < 2:
                continue

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
                acc = 100 * correct / total if total > 0 else 0
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {acc:.2f}%')

        if len(train_loader) > 0:
            avg_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total if total > 0 else 0
            print(f'Epoch [{epoch + 1}/{num_epochs}] 平均损失: {avg_loss:.4f}, 准确率: {epoch_acc:.2f}%')

    print("训练完成!")

    # 测试模型
    validator = FaceValidator(model, device, class_names)
    validator.test_accuracy(test_loader)

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_names': class_names,
    }, 'chinese_face_model.pth')
    print(f"\n模型已保存!")


if __name__ == "__main__":
    main()