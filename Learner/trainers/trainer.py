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
import warnings


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
        use_fp16=True,  # 启用FP16与FP32混合精度训练，进行梯度缩放
    ):
        self.batch_size = batch_size
        self.model_path = model_path
        self.use_fp16 = False

        # -------------------------- 检查 GPU 是否可用 --------------------------
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # 如果有可用的 GPU，则使用 GPU
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
            self.use_fp16 = use_fp16  # 若GPU可用则同意混合精度
        else:
            self.device = torch.device("cpu")  # 如果没有可用的 GPU，则使用 CPU
            print("使用 CPU")

        # -------------------------- 梯度缩放器 --------------------------
        self.scaler = GradScaler() if self.use_fp16 else None  # 初始化梯度缩放器
        self._autocast_warning_issued = False  # 标记是否已发出警告（避免重复）

        # -------------------------- 确保模型目录存在 --------------------------
        if not os.path.exists(self.model_path):
            print(f"创建模型保存目录: {self.model_path}")
            os.makedirs(self.model_path, exist_ok=True)

        # -------------------------- 模型 --------------------------
        self.model = model.to(self.device)

        # -------------------------- 损失函数 --------------------------
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        # -------------------------- 训练集和发展集加载器 --------------------------
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader

        # -------------------------- 优化器 --------------------------
        if optimizer is None:
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
            )  # 默认使用 SGD 优化器
        else:
            self.optimizer = optimizer

        # -------------------------- 学习率调度器 --------------------------
        if scheduler is None:
            self.scheduler = WarmUpCosineAnnealingLR(
                self.optimizer,
                warmup_steps=5,
                total_steps=total_epochs,
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
        # patience=10  # 早停耐心值
    ):
        """模型训练"""
        # -------------------------- 加载预训练模型 --------------------------
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
            self.scheduler.load_state_dict(
                checkpoint['scheduler_state_dict'])  # 加载学习率调度器状态
            # 获取之前训练的轮数（理论上 start_epoch == pretrain）
            start_epoch = checkpoint['epoch'][-1]
            pre_loss = checkpoint['train_loss'][-1]  # 获取之前训练的损失

            print(
                f'加载预训练模型: {pretrain}，已训练轮数: {start_epoch}，先前训练集损失: {pre_loss}')

            if self.dev_dataloader is not None and 'dev_metrics' in checkpoint:
                self.dev_loss = checkpoint['dev_loss'][-1]
                pre_acc = checkpoint['dev_score'][-1]  # 获取之前训练的准确率
                print(f'先前发展集损失: {self.dev_loss}，先前发展集准确率: {pre_acc}')

        else:
            print('无预训练模型，从零开始训练 . . .')
            start_epoch = 0

        # -------------------------- 迭代训练 --------------------------
        epoch_list = []
        train_loss_list = []  # 训练集损失列表
        dev_loss_list = []  # 发展集损失列表
        dev_metrics_list = []  # 发展集评分列表

        # best_dev_loss = float('inf')
        # epochs_no_improve = 0

        for epoch in range(epoch_num):
            current_epoch = start_epoch + epoch + 1
            print(
                f"\n===== 第 {current_epoch}/{start_epoch + epoch_num} 轮训练 =====")

            # 遍历训练集，训练模型参数
            self.model.train()  # 设置模型为训练模式
            train_loss = 0.0  # 训练集损失
            # train_batch_size_list = []  # 每个批次的样本数
            # 获取每个 batch 的 loss
            for loss, batch_size in self.train():
                train_loss += loss.data.item()  # 获取损失值并求和

            # 存在发展集
            if self.dev_dataloader is not None and 'dev_metrics' in checkpoint:
                # 计算发展集上的损失值
                self.model.eval()  # 设置模型为评估模式
                dev_loss = 0.0  # 发展集损失
                metrics_list = []  # 每个批次的分数列表
                dev_batch_size_list = []  # 每个批次的样本数
                with torch.no_grad():    # 禁用梯度计算（节省显存+提速）
                    # 计算发展集上的损失值和准确度
                    for loss, score, batch_size in self.dev():
                        dev_loss += loss.data.item()
                        metrics_list.append(score)
                        dev_batch_size_list.append(batch_size)

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

            train_loss = train_loss / \
                len(self.train_dataloader)  # 训练集每个样本的平均损失

            if self.dev_dataloader is not None:
                # 验证集每个样本的平均损失
                dev_loss = dev_loss / \
                    len(self.dev_dataloader)

                # -------------------------- 计算数据集整体分数 --------------------------
                total_samples = sum(dev_batch_size_list)  # 数据集总样本数
                if total_samples == 0:
                    dev_metrics = 0.0  # 避免除以0
                else:
                    # 转换为 numpy 数组计算（更高效）
                    metrics_array = np.array(metrics_list)
                    batch_size_array = np.array(dev_batch_size_list)
                    dev_metrics = np.sum(
                        metrics_array * batch_size_array) / total_samples
                # ------------------------------------------------------------------------

                print(
                    f'第 {start_epoch + epoch + 1} 轮训练结束，训练集 loss 为 {train_loss}，发展集 loss 为 {dev_loss}，发展集评分为 {dev_metrics}')
                train_loss_list.append(train_loss)
                dev_loss_list.append(dev_loss)
                dev_metrics_list.append(dev_metrics)
                epoch_list.append(start_epoch + epoch + 1)

            else:
                print(
                    f'第 {start_epoch + epoch + 1} 轮训练结束，训练集 loss 为 {train_loss}')
                train_loss_list.append(train_loss)
                epoch_list.append(start_epoch + epoch + 1)

        # -------------------------- 保存模型和优化器状态 --------------------------
        if self.dev_dataloader is not None:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),  # 模型权重
                'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器状态：参数梯度、动量等信息
                'scheduler_state_dict': self.scheduler.state_dict(),  # 学习率调度器状态：已训练步数、预热进度、衰减系数等
                # 本次训练的轮数列表
                'epoch': list(range(pretrain + 1, pretrain + epoch_num + 1)),
                'train_loss': train_loss_list,  # 训练集损失列表
                'dev_loss': dev_loss_list,  # 验证集损失列表
                'dev_metrics': dev_metrics_list,  #
            }
        else:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': list(range(pretrain + 1, pretrain + epoch_num + 1)),
                'train_loss': train_loss_list,
            }

        # 保存模型检查点
        print(f'保存模型到 {self.model_path}checkpoint_{pretrain + epoch_num}.pth')
        torch.save(checkpoint, self.model_path +
                   f'checkpoint_{pretrain + epoch_num}.pth')  # 保存模型

    def forward_pass(self, batch, is_training=True):
        """模型前向传播与损失计算"""
        # TODO: 需重写方法，视任务而定
        # 数据移到GPU/CPU
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # 训练模式才清空梯度，验证模式跳过
        if is_training:
            self.optimizer.zero_grad()

        # 混合精度前向传播（FP16）
        with autocast(enabled=self.use_fp16, device_type=self.device.type):
            outputs = outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits  # 预测分数（供后续计算准确率）

            self._check_autocast_usage(loss)

        return loss, logits, labels.size(0)  # 当前批次样本数

    def _check_autocast_usage(self, loss: torch.Tensor):
        """检测：use_fp16=True 时，前向传播是否启用了 autocast"""
        if not self.use_fp16:
            return  # 未开启混合精度，无需检测

        # 关键：通过 loss 的精度判断前向是否用了 autocast（FP16 前向的 loss 是 FP16 张量）
        if loss.dtype == torch.float32:
            # 仅警告一次，避免每个批次都刷屏
            if not self._autocast_warning_issued:
                warnings.warn(
                    "\n⚠️  警告：已开启 use_fp16=True（混合精度训练），但前向传播（forward_pass）未添加 autocast 上下文！"
                    "\n   后果：混合精度训练失效（仍用 FP32），梯度缩放（GradScaler）可能导致训练震荡/崩溃。"
                    "\n   修复方案：在 forward_pass 的前向传播代码外包裹："
                    "\n   with autocast(enabled=self.use_fp16):"
                    "\n       outputs = self.model(...)"
                    "\n       loss = outputs.loss",
                    UserWarning,
                    stacklevel=2
                )
                self._autocast_warning_issued = True  # 标记已警告

    def parameter_update(self, loss: torch.Tensor):
        """模型参数更新方法，用于在训练方法内部前向传播之后调用"""
        if self.use_fp16:
            # 混合精度
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 普通精度
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)  # 裁剪梯度范数≤1.0，防止梯度爆炸
            self.optimizer.step()  # 更新优化器状态

        self.scheduler.step()  # 更新学习率

    def train(self):
        """生成器：在训练集上个更新模型权重"""
        # TODO: 需重写方法，视任务而定
        for batch in tqdm(self.train_dataloader, desc="模型训练"):
            # 1. 前向流程：清空梯度→前向传播→计算损失
            loss, _, batch_size = self.forward_pass(batch, is_training=True)

            # 2. 更新流程：反向传播→参数更新
            self.parameter_update(loss)

            # 3. 生成当前批次的 loss 和 batch_size（供外层统计总损失）
            yield loss, batch_size

    def compute_dev_metrics(self, logits: torch.Tensor, labels: torch.Tensor):
        """计算验证集指标"""
        # TODO: 需重写方法，视任务而定
        # 计算预测类别（取logits最大值索引）
        preds = torch.argmax(logits, dim=1)
        # 转CPU+detach（脱离计算图），避免显存泄漏
        preds_np = preds.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        # 计算准确率（与原有逻辑完全一致）
        return accuracy_score(labels_np, preds_np)

    def dev(self):
        """在发展集上验证模型，并更新学习率"""
        # TODO: 需重写方法，视任务而定
        for batch in tqdm(self.dev_dataloader, desc="模型验证"):
            # 第一步：调用forward_pass，获取loss、logits、batch_size（验证模式：is_training=False）
            loss, logits, batch_size = self.forward_pass(
                batch, is_training=False)

            # 第二步：调用compute_dev_metrics，计算当前批次准确率
            # labels从原始batch取（已在forward_pass移到设备）
            acc = self.compute_dev_metrics(logits, batch["labels"])

            # 第三步：逐批次yield结果（与原有接口完全兼容）
            yield loss, acc, batch_size

    def predict(self, test_dataloader):
        '''模型推理'''
        # TODO: 需重写方法，视任务而定

    def eval(self, test_dataloader):
        '''模型评估'''
        # TODO: 需重写方法，视任务而定
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
