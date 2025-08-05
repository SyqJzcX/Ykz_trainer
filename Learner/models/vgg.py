import torch
from torch import nn

# 增加非线性：每个卷积层后面都会跟随一个非线性激活函数（通常是ReLU），堆叠多个卷积层可以增加网络的非线性能力，使得网络能够学习更复杂的特征表示。
# 减少参数数量：通过堆叠多个小卷积核（如3x3）代替大卷积核（如5x5或7x7），可以在保持相同感受野的情况下减少模型的参数数量。例如，三个3x3卷积核的参数总量少于一个7x7卷积核的参数总量，这有助于减轻过拟合并减少计算量。
# 保持图像性质：小卷积核有利于更好地保持图像的细节和空间性质，因为它们具有较小的 Receptive Field（感受野），可以捕捉到更多的局部特征。
# 提升特征学习能力：多层卷积层可以捕获不同尺度的特征，增强了网络对特征的学习能力。这种深度的增加有助于网络学习到更抽象的特征表示，从而提高分类的准确性。
# 保持特征图尺寸：在VGG网络中，卷积层（使用padding=1）保持了输入特征图的尺寸不变，直到池化层才将特征图尺寸减半。这种设计使得网络可以在较深的层次上保持较高的分辨率，有助于捕捉更多的空间信息。
# 可以通过平均池化层代替全连接层，减小参数数量，提升泛化能力

class VGG(nn.Module):
    """视觉几何群网络"""
    def __init__(self):
        super(VGG, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # 32x32x3 ==> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),  # 32x32x64 ==> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x16x64
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  # 16x16x64 ==> 16x16x128
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),  # 16x16x128 ==> 16x16x128
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x8x128
            nn.Dropout2d(0.1)  # 池化后随机丢弃特征图
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),  # 8x8x128 ==> 8x8x256
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),  # 8x8x256 ==> 8x8x256
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),  # 8x8x256 ==> 8x8x256
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 4x4x256
            nn.Dropout2d(0.2),  # 仅保留一个卷积后Dropout
        )
        self.mid = nn.Sequential(
            nn.AdaptiveAvgPool2d(
                (2, 2)  # 输出图像尺寸，一般比输入图像小效果好一些
                ),  # 自适应平均池化层，一般置于卷积层和全连接层之间
            nn.Flatten(),
            nn.Dropout(0.3),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 2 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 10)
        )
    
    def forward(self, input):
        feature = self.block1(input)
        feature = self.block2(feature)
        feature = self.block3(feature)
        hidden = self.mid(feature)
        hidden = self.fc1(hidden)
        out = self.fc2(hidden)
        return out
    