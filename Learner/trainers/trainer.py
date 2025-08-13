import torch
from torch import nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from tqdm import tqdm
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from Learner.utils.lr import WarmUpCosineAnnealingLR


class Trainer:
    """模型训练类"""

    def __init__(
        self,
        model,  # 模型
        train_dataloader=None,  # 训练集加载器
        dev_dataloader=None,  # 发展集加载器：有无发展集均可
        criterion=None,  # 损失函数
        optimizer=None,  # 优化器
        scheduler=None,  # 学习率调度器
        batch_size=512,  # 样本批量
        total_epochs=50,  # 预期总训练轮数
        model_path='./models/checkpoints/',  # 模型检查点保存路径
    ):
        self.batch_size = batch_size
        self.model_path = model_path

        # 检查 GPU 是否可用
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # 如果有可用的 GPU，则使用 GPU
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")  # 如果没有可用的 GPU，则使用 CPU
            print("使用 CPU")

        # 模型
        self.model = model.to(self.device)

        # 损失函数
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        # 训练集和发展集加载器
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader

        # 优化器
        if optimizer is None:
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
            )  # 默认使用 SGD 优化器
        else:
            self.optimizer = optimizer

        # 学习率调度器
        if scheduler is None:
            self.scheduler = WarmUpCosineAnnealingLR(
                self.optimizer,
                warmup_epochs=5,
                total_epochs=total_epochs,
                warmup_start_factor=0.01,
                warmup_end_factor=1.0,
                cos_end_factor=0.01
            )
        else:
            self.scheduler = scheduler

    def fit(
        self,
        epoch_num,  # 训练轮次数
        pretrain=0,  # 预训练模型编号（0 代表没有）
        patience=10  # 早停耐心值
    ):
        """模型训练"""
        # 加载预训练模型
        if pretrain:
            checkpoint = torch.load(  # 加载保存的检查点
                self.model_path + f'checkpoint_{pretrain}.pth',
                map_location=self.device,
                weights_only=False
            )

            self.model.load_state_dict(
                checkpoint['model_state_dict'])  # 加载模型参数
            self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])  # 加载优化器状态
            # 获取之前训练的轮数（理论上 start_epoch == pretrain）
            start_epoch = checkpoint['epoch'][-1]
            pre_loss = checkpoint['train_loss'][-1]  # 获取之前训练的损失

            print(
                f'加载预训练模型: {pretrain}，已训练轮数: {start_epoch}，先前训练集损失: {pre_loss}')

            if self.dev_dataloader is not None:
                self.dev_loss = checkpoint['dev_loss'][-1]
                pre_acc = checkpoint['dev_acc'][-1]  # 获取之前训练的准确率
                print(f'先前发展集损失: {self.dev_loss}，先前发展集准确率: {pre_acc}')

        else:
            print('无预训练模型，从零开始训练 . . .')
            start_epoch = 0

        # 迭代训练
        epoch_list = []
        train_loss_list = []  # 训练集损失列表
        dev_loss_list = []  # 发展集损失列表
        dev_acc_list = []  # 发展集准确率列表

        best_dev_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epoch_num):
            # 该epoch的
            train_loss = 0.0  # 训练集损失

            # 遍历训练集，训练模型参数
            self.model.train()  # 设置模型为训练模式
            # 获取每个 batch 的 loss
            for loss in self.train_model():
                train_loss += loss.data.item()  # 获取损失值并求和

            # 存在发展集
            if self.dev_dataloader is not None:
                # 计算发展集上的损失值
                self.model.eval()  # 设置模型为评估模式
                with torch.no_grad():  # 禁用梯度计算以提高性能
                    # 计算发展集上的损失值和准确度
                    dev_loss = 0.0  # 发展集损失
                    acc_list = []
                    for loss, acc in self.dev_model():
                        dev_loss += loss.data.item()
                        acc_list.append(acc)

                # 早停检查
                # if dev_loss < best_dev_loss:
                #     best_dev_loss = dev_loss
                #     epochs_no_improve = 0
                #     # 保存最佳模型
                #     torch.save(self.model.state_dict(), f"{self.model_path}best_model.pth")
                # else:
                #     epochs_no_improve += 1
                #     if epochs_no_improve >= patience:
                #         print(f"早停触发: 连续 {patience} 轮验证损失未改善")
                #         break

            # 更新学习率并监测验证集上的性能
            self.scheduler.step()

            train_loss = train_loss / \
                len(self.train_dataloader)  # 训练集每个样本的平均损失

            if self.dev_dataloader is not None:
                dev_loss = dev_loss / \
                    len(self.dev_dataloader)  # 验证集每个样本的平均损失
                dev_acc = np.mean(np.array(acc_list))
                print(
                    f'第 {start_epoch + epoch + 1} 轮训练结束，训练集 loss 为 {train_loss}，发展集 loss 为 {dev_loss}，发展集准确率为 {dev_acc}')
                train_loss_list.append(train_loss)
                dev_loss_list.append(dev_loss)
                dev_acc_list.append(dev_acc)
                epoch_list.append(start_epoch + epoch + 1)

            else:
                print(
                    f'第 {start_epoch + epoch + 1} 轮训练结束，训练集 loss 为 {train_loss}')
                train_loss_list.append(train_loss)
                epoch_list.append(start_epoch + epoch + 1)

        if self.dev_dataloader is not None:
            # 保存模型和优化器状态
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 当前训练的轮数列表
                'epoch': list(range(pretrain + 1, pretrain + epoch_num + 1)),
                'train_loss': train_loss_list,
                'dev_loss': dev_loss_list,
                'dev_acc': dev_acc_list,
            }
        else:
            # 保存模型和优化器状态
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 当前训练的轮数列表
                'epoch': list(range(pretrain + 1, pretrain + epoch_num + 1)),
                'train_loss': train_loss_list,
            }

        # 保存模型检查点
        print(f'保存模型到 {self.model_path}checkpoint_{pretrain + epoch_num}.pth')
        torch.save(checkpoint, self.model_path +
                   f'checkpoint_{pretrain + epoch_num}.pth')  # 保存模型

    def train_model(self):
        """在训练集上个更新模型权重"""
        # TODO: 需要根据不同模型重新定义
        for data, label in tqdm(self.train_dataloader, desc="模型训练"):
            data = data.to(self.device)
            label = label.to(self.device)

            out = self.model(data)  # 输出
            loss = self.criterion(out, label)  # 损失

            self.optimizer.zero_grad()  # 将梯度置零放在循环开始处，以避免潜在的优化问题
            loss.backward()  # 反向传播
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)  # 裁剪梯度范数，防止梯度爆炸
            self.optimizer.step()  # 更新参数

            yield loss

    def dev_model(self):
        """在发展集上验证模型，并更新学习率"""
        # TODO: 需要根据不同模型重新定义
        for data, label in tqdm(self.dev_dataloader, desc="模型验证"):
            data = data.to(self.device)
            label = label.to(self.device)

            out = self.model(data)  # 输出

            loss = self.criterion(out, label)  # 损失
            preds = torch.argmax(out, dim=1)  # 预测类别

            acc = accuracy_score(label.cpu(), preds.cpu())  # 计算准确率

            yield loss, acc

    def predict(self, test_dataloader):
        '''模型推理'''

    def eval(self, test_dataloader):
        '''模型评估'''
        # TODO: 需要根据不同模型重新定义
        self.model.eval()  # 设置模型为评估模式

        # 计算测试集上的准确度
        test_true = []
        test_pred = []
        for data, label in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():  # 评估模式，不计算梯度，节省内存
                out = self.model(data)  # 输出

            # print(out.shape)  # 输出形状
            # print(label.shape)  # 标签形状

            pred = torch.argmax(out, dim=1)  # 预测类别
            # lbl = torch.argmax(label, dim=1)  # 实际类别
            test_true.extend(label.cpu())  # 放入列表末尾
            test_pred.extend(pred.cpu())

        # 计算评估指标
        return {
            "accuracy_score": accuracy_score(test_true, test_pred),
            "precision_score": precision_score(test_true, test_pred, average='macro'),
            "recall_score": recall_score(test_true, test_pred, average='macro'),
            "f1_score": f1_score(test_true, test_pred, average='macro'),
            "confusion_matrix": confusion_matrix(test_true, test_pred)
        }
