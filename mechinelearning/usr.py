import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 首先需要定义你的模型类（必须与训练时完全相同）
class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNeuralNet, self).__init__()
        # 这里需要填写你实际的模型结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # CIFAR10经过两次池化后是8x8
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_and_test_model():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 使用 weights_only=False 加载模型
        model = torch.load('model.pth', map_location=device, weights_only=False)
        print("✅ 成功加载整个模型")
        
        # 确保模型在评估模式
        model.eval()
        print("模型已设置为评估模式")
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None
    
    # 后续代码与之前相同...
    # CIFAR10数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载CIFAR10测试数据
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 测试准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'📊 模型准确率: {accuracy:.2f}%')
    
    return model

if __name__ == "__main__":
    model = load_and_test_model()
