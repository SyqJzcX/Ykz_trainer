from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler, SequentialLR
import math
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import matplotlib.pyplot as plt


def WarmUpCosineAnnealingLR(
    optimizer,
    warmup_steps=5,  # 预热轮数
    total_steps=50,  # 总训练轮数
    warmup_start_factor=0.001,  # 预热初始学习率因子
    warmup_end_factor=1.0,  # 预热结束学习率因子
    cos_end_factor=0.001  # 最终学习率因子
):
    """线性预热+余弦退火学习率调度器"""
    # 获取基础学习率
    base_lr = optimizer.param_groups[0]['lr']

    # 线性预热调度器：从0.0到base_lr，共warmup_epochs步
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,  # 初始学习率 = base_lr * start_factor = 0.001 * 0.01
        end_factor=warmup_end_factor,  # 结束学习率 = base_lr * end_factor = 0.001 * 1.0
        total_iters=warmup_steps  # 预热步数（按epoch）
    )

    # 余弦退火调度器：预热结束后开始退火
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_steps - warmup_steps),  # 退火周期（总步数）
        eta_min=base_lr * cos_end_factor  # 最小学习率（可选）
    )

    # 组合两个调度器：先执行预热，再执行退火
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]  # 预热结束的步数（切换到退火）
    )


class LinearDecayWithWarmup(_LRScheduler):
    """线性预热+线性退火学习率调度器"""
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1  # 步数从1开始计数
        if current_step <= self.warmup_steps:
            # 预热阶段：线性增长
            lr_factor = current_step / self.warmup_steps
        else:
            # 衰减阶段：线性下降
            lr_factor = 1 - (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        # 确保学习率非负
        return [base_lr * max(lr_factor, 0.0) for base_lr in self.base_lrs]


def WarmUpInverseSqrtLR(
    optimizer,
    d_model: int = 512,  # Transformer模型维度（词嵌入/隐藏层维度）
    warmup_epochs: int = 5,  # 预热轮数
    total_epochs: int = 50,  # 总训练轮数
    # 基础学习率因子（最终base_lr = base_lr_factor / sqrt(d_model)）
    base_lr_factor: float = 1.0,
    min_lr_factor: float = 1e-4  # 最小学习率因子（避免学习率过低）
):
    """
    专为Transformer设计的NOAM学习率调度器：线性预热 + 逆平方根衰减
    核心逻辑遵循Transformer原论文，学习率与模型维度d_model强相关，确保训练稳定性

    Args:
        optimizer: PyTorch优化器（如Adam、AdamW）
        d_model: Transformer模型的维度（词嵌入维度或隐藏层维度，如512/768）
        warmup_epochs: 线性预热轮数（通常为总轮数的5%~10%）
        total_epochs: 总训练轮数
        base_lr_factor: 基础学习率因子（原论文中默认1.0，最终基础LR = base_lr_factor / sqrt(d_model)）
        min_lr_factor: 最小学习率因子（衰减阶段学习率不低于此值 × 基础LR，防止梯度消失）
    """
    # 计算NOAM基础学习率（与模型维度d_model绑定，核心设计）
    base_lr = base_lr_factor / math.sqrt(d_model)
    # 更新优化器的基础学习率（覆盖原optimizer的lr设置，确保NOAM缩放逻辑生效）
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr

    # -------------------------- 1. 线性预热调度器 --------------------------
    # 预热阶段：学习率从 (base_lr × 1/sqrt(warmup_epochs)) 线性增长到 base_lr
    # 符合原论文预热公式：lr = base_lr × step / warmup_steps^(1.5)
    class WarmupLinearLR(_LRScheduler):
        def __init__(self, optimizer, warmup_epochs, base_lr, last_epoch=-1):
            self.warmup_epochs = warmup_epochs
            self.base_lr = base_lr
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            # 当前轮数（从1开始计数，避免step=0时分母为0）
            current_epoch = self.last_epoch + 1
            # 预热公式：线性增长，current_epoch=1时lr=base_lr/sqrt(warmup_epochs)，current_epoch=warmup_epochs时lr=base_lr
            warmup_lr = self.base_lr * current_epoch / \
                math.sqrt(self.warmup_epochs)
            return [warmup_lr for _ in self.base_lrs]

    warmup_scheduler = WarmupLinearLR(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        base_lr=base_lr
    )

    # -------------------------- 2. 逆平方根衰减调度器 --------------------------
    # 衰减阶段：学习率从base_lr按 1/sqrt(current_epoch) 衰减，不低于 min_lr
    class InverseSqrtLR(_LRScheduler):
        def __init__(self, optimizer, warmup_epochs, base_lr, min_lr_factor, last_epoch=-1):
            self.warmup_epochs = warmup_epochs
            self.base_lr = base_lr
            self.min_lr = base_lr * min_lr_factor  # 最小学习率
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            # 当前轮数（包含预热轮数，从warmup_epochs+1开始计数）
            current_epoch = self.last_epoch + 1 + self.warmup_epochs
            # 逆平方根衰减公式：lr = base_lr / sqrt(current_epoch)
            decay_lr = self.base_lr / math.sqrt(current_epoch)
            # 确保学习率不低于最小值
            return [max(decay_lr, self.min_lr) for _ in self.base_lrs]

    decay_scheduler = InverseSqrtLR(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        base_lr=base_lr,
        min_lr_factor=min_lr_factor
    )

    # -------------------------- 3. 组合调度器（先预热，后衰减） --------------------------
    return SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_epochs]  # 预热结束后切换到衰减调度器
    )
