import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import matplotlib.pyplot as plt


def WarmUpCosineAnnealingLR(
    optimizer,
    warmup_epochs=5,  # 预热轮数
    total_epochs=50,  # 总训练轮数
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
        total_iters=warmup_epochs  # 预热步数（按epoch）
    )

    # 余弦退火调度器：预热结束后开始退火
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_epochs - warmup_epochs),  # 退火周期（总步数）
        eta_min=base_lr * cos_end_factor  # 最小学习率（可选）
    )
    
    # 组合两个调度器：先执行预热，再执行退火
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]  # 预热结束的步数（切换到退火）
    )
    
def draw_lr(
    optimizer,
    scheduler,
    total_epochs=50  # 总训练轮数
):
    """绘制学习率调度器的学习率变化曲线"""
    # 绘制学习率曲线
    lrs = []
    for epoch in range(total_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        print(f"Epoch {epoch+1}: LR = {current_lr:.6f}")
        scheduler.step()  # 每个epoch更新一次学习率

    # 可视化（可选）
    plt.plot(range(1, total_epochs+1), lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("SequentialLR: Warmup + Cosine Annealing")
    plt.show()
    
    return lrs


if __name__ == "__main__":
    # 示例用法
    model = torch.nn.Linear(10, 1)  # 简单模型
    optimizer = SGD(model.parameters(), lr=0.01)  # 优化器

    # 创建调度器
    scheduler = WarmUpCosineAnnealingLR(
        optimizer,
        warmup_epochs=5,
        total_epochs=50,
        warmup_start_factor=0.001,
        warmup_end_factor=1.0,
        cos_end_factor=0.001
    )

    # 绘制学习率曲线
    draw_lr(optimizer, scheduler, total_epochs=30)
