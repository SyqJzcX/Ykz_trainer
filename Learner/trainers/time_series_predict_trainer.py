from Learner.trainers.trainer import Trainer
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from Learner.utils.plot import plot_forecast_comparison


class TimeSeriesPredictTrainer(Trainer):
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
        super(TimeSeriesPredictTrainer, self).__init__(
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

            # 转换为numpy并展平（单个批次的所有时间点）
            preds_np = pred.cpu().numpy().flatten()  # 形状：(256*2,) = (512,)
            labels_np = label.cpu().numpy().flatten()

            # 计算当前批次的R²（越大越好）
            r2 = r2_score(labels_np, preds_np)

            yield loss, r2

    def eval(self, test_dataloader):
        '''模型评估'''
        self.model.eval()  # 设置模型为评估模式

        # 计算测试集上的准确度
        test_true = []
        test_pred = []
        for data, label in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
            data = data.to(self.device, dtype=torch.float32)
            label = label.to(self.device, dtype=torch.float32)

            # 提取目标特征列
            label = label[:, :, self.label_idx].unsqueeze(-1)

            with torch.no_grad():  # 评估模式，不计算梯度，节省内存
                out = self.model(data)  # 输出

            pred = out[:, :, self.label_idx].unsqueeze(-1)  # 预测

            test_pred.extend(pred.cpu().numpy().flatten())  # 放入列表末尾
            test_true.extend(label.cpu().numpy().flatten())

        plot_forecast_comparison(
            original_series=test_true, predicted_series=test_pred)
        
        print(len(test_pred))
        print(len(test_true))

        # 计算评估指标
        return {
            "mean_squared_error": mean_squared_error(test_true, test_pred),
            "root_mean_squared_error": np.sqrt(mean_squared_error(test_true, test_pred)),
            "mean_absolute_error": mean_absolute_error(test_true, test_pred),
            "mean_absolute_percentage_error(%)": mean_absolute_percentage_error(test_true, test_pred) * 100,
            "r2_score": r2_score(test_true, test_pred)
        }
