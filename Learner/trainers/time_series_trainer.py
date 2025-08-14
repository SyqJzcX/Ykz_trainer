from learner.trainers.trainer import Trainer
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score


class TimeSeriesTrainer(Trainer):
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
        label_idx=0  # 指定需要预测的特征索引（0-6）
    ):
        super(TimeSeriesTrainer, self).__init__(
            model,  # 模型
            train_dataloader=train_dataloader,  # 训练集加载器
            dev_dataloader=dev_dataloader,  # 发展集加载器：有无发展集均可
            criterion=criterion,  # 损失函数
            optimizer=optimizer,  # 优化器
            scheduler=scheduler,  # 学习率调度器
            batch_size=batch_size,  # 样本批量
            total_epochs=total_epochs,  # 预期总训练轮数
            model_path=model_path,  # 模型检查点保存路径
        )
        self.label_idx = label_idx

    def train_model(self):
        """在训练集上个更新模型权重"""
        for data, label in tqdm(self.train_dataloader, desc="模型训练"):
            data = data.to(self.device, dtype=torch.float32)
            label = label.to(self.device, dtype=torch.float32)

            # 只提取需要预测的特征列（标签形状：[256,2,7] → [256,2,1]）
            label = label[:, :, self.label_idx].unsqueeze(-1)

            # 前向传播
            out = self.model(data)  # 模型输出：[256,2,7]
            pred = out[:, :, self.label_idx].unsqueeze(-1)  # 只取预测列：[256,2,1]

            # 计算损失（仅用指定特征列）
            loss = self.criterion(pred, label)

            self.optimizer.zero_grad()  # 将梯度置零放在循环开始处，以避免潜在的优化问题
            loss.backward()  # 反向传播
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)  # 裁剪梯度范数，防止梯度爆炸
            self.optimizer.step()  # 更新参数

            yield loss

    def dev_model(self):
        """在发展集上验证模型，并更新学习率"""
        for data, label in tqdm(self.dev_dataloader, desc="模型验证"):
            data = data.to(self.device, dtype=torch.float32)
            label = label.to(self.device, dtype=torch.float32)

            # 提取目标特征列
            label = label[:, :, self.label_idx].unsqueeze(-1)

            # 前向传播
            out = self.model(data)
            pred = out[:, :, self.label_idx].unsqueeze(-1)

            # 损失
            loss = self.criterion(pred, label)
            preds = torch.argmax(out, dim=1)  # 预测类别
            
            # 转换为numpy并展平（单个批次的所有时间点）
            preds_np = pred.cpu().numpy().flatten()  # 形状：(256*2,) = (512,)
            labels_np = label.cpu().numpy().flatten()

            # 计算当前批次的R²（越大越好）
            r2 = r2_score(labels_np, preds_np)

            yield loss, r2

    def eval(self, test_dataloader):
        '''模型评估'''
