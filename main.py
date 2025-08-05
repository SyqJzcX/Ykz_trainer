import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from Learner.trainer import Trainer
import torch.optim as optim
from torch import nn
from Learner.models.vgg import VGG
from Learner.utils.lr import WarmUpCosineAnnealingLR, draw_lr
from Learner.utils.plot import draw_loss, conf_matrix


def train():
    # 实例化VGG模型
    model = VGG()

    # 超参数
    batch_size = 256  # 批量大小
    epoch_num = 50  # 训练轮数
    learning_rate = 0.01  # 学习率

    warmup_epochs = 5
    warmup_start_factor = 0.01
    warmup_end_factor = 1.0
    cos_end_factor = 0.01

    # 数据预处理和加载
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        transforms.ToTensor(),  # 转换为Tensor（0-1范围）
        # transforms.Normalize(  # 标准化到均值0、方差1
        #     mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10的均值
        #     std=[0.2470, 0.2435, 0.2616]  # CIFAR-10的标准差
        # )
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor（0-1范围）
        # transforms.Normalize(  # 标准化到均值0、方差1
        #     mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10的均值
        #     std=[0.2470, 0.2435, 0.2616]  # CIFAR-10的标准差
        # )
    ])

    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,  # 训练集
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,  # 测试集
        download=True,  # 下载数据集
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱顺序
        num_workers=2  # 多进程加载（根据CPU核心数调整）
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集无需打乱
        num_workers=2
    )

    # 模型训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.1,
    #     patience=5
    # )
    scheduler = WarmUpCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epoch_num,
        warmup_start_factor=warmup_start_factor,
        warmup_end_factor=warmup_end_factor,
        cos_end_factor=cos_end_factor
    )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        dev_dataloader=test_loader,  # 使用测试集作为发展集
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=batch_size,
        model_path="./models/checkpoints/"  # 模型保存路径
    )

    trainer.fit(
        epoch_num=epoch_num,  # 训练轮次
        pretrain=0,  # 是否使用预训练参数
    )

    draw_loss("./models/checkpoints/")  # 绘制训练损失和准确率曲线


def eval():
    # 实例化VGG模型
    model = VGG()

    checkpoint = torch.load(  # 加载保存的检查点
        './models/checkpoints/checkpoint_50.pth',
        weights_only=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数

    # 数据预处理和加载
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor（0-1范围）
        # transforms.Normalize(  # 标准化到均值0、方差1
        #     mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10的均值
        #     std=[0.2470, 0.2435, 0.2616]  # CIFAR-10的标准差
        # )
    ])

    batch_size = 256  # 批量大小

    # 加载CIFAR-10数据集
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,  # 训练集
        download=True,
        transform=test_transform
    )

    classes = test_dataset.classes
    print(classes)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集无需打乱
        num_workers=2
    )

    trainer = Trainer(
        model=model,
        # train_dataloader=train_loader,
        # dev_dataloader=test_loader,  # 使用测试集作为发展集
        # criterion=criterion,
        # optimizer=optimizer,
        # scheduler=scheduler,
        # batch_size=batch_size,
        # model_path="./models/checkpoints/"  # 模型保存路径
    )

    # 模型评估
    eva = trainer.eval(test_loader)
    
    print(eva["accuracy_score"])

    conf_matrix(eva["confusion_matrix"])


if __name__ == "__main__":
    # train()
    eval()
