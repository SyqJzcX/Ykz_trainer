import os
import torch
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.colors import get_named_colors_mapping

# 设置中文显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def draw_loss(
    model_path
):
    '''绘制训练曲线（有无发展集数据均可）'''
    # 查找目标目录下所有符合条件的.pth文件
    file_pattern = os.path.join(model_path, 'checkpoint_*.pth')
    pth_files = glob(file_pattern)

    # 初始化列表用于存储各指标数据
    all_epochs = []
    all_train_losses = []
    all_dev_losses = []
    all_dev_scores = []

    # 标志位：是否包含验证集指标（初始为None）
    has_dev_metrics = None

    # 遍历找到的文件
    for file_path in pth_files:
        try:
            checkpoint = torch.load(file_path, weights_only=False)

            all_epochs.extend(checkpoint['epoch'])
            all_train_losses.extend(checkpoint['train_loss'])

            # 如果是第一个有效文件，确定是否包含验证集指标
            if has_dev_metrics is None:
                has_dev_metrics = (
                    'dev_loss' in checkpoint and 'dev_score' in checkpoint)
                if has_dev_metrics:
                    print("检测到验证集指标，将绘制完整曲线")
                else:
                    print("未检测到验证集指标，仅绘制训练损失曲线")

            # 如果包含验证集指标，收集验证数据
            if has_dev_metrics:
                all_dev_losses.extend(checkpoint['dev_loss'])
                all_dev_scores.extend(checkpoint['dev_score'])

            print(file_path)

        except Exception as e:
            print(f"读取 {file_path} 时出错，错误信息: {e}")

    # 对数据按epoch排序（处理可能的乱序情况）
    if has_dev_metrics:
        # 对数据按epoch排序
        sorted_data = sorted(zip(all_epochs, all_train_losses,
                                 all_dev_losses, all_dev_scores), key=lambda x: x[0])
        epochs, train_losses, dev_losses, dev_accuracies = zip(
            *sorted_data)

    else:
        sorted_data = sorted(
            zip(all_epochs, all_train_losses), key=lambda x: x[0])
        epochs, train_losses = zip(*sorted_data)

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 绘制损失曲线（左Y轴）
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses,
             color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 如果有验证集指标
    if has_dev_metrics:
        # 绘制验证损失曲线（左Y轴）
        ax1.plot(epochs, dev_losses,
                 color='tab:orange', label='Validation Loss')

        # 绘制准确率曲线（右Y轴）
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Accuracy (%)', color=color)
        ax2.plot(epochs, [acc * 100 for acc in dev_accuracies],
                 color=color, label='Validation Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        # 确保准确率轴范围在0-100%之间，同时留出适当边距
        min_acc = min(dev_accuracies) * 100
        max_acc = max(dev_accuracies) * 100
        margin = max(5, (max_acc - min_acc) * 0.1)  # 边距至少为5%或数据范围的10%
        ax2.set_ylim(max(0, min_acc - margin), min(100, max_acc + margin))

        # 添加图例和标题
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.title('Model Training Metrics Over Epochs')

    else:
        # 仅显示训练损失
        ax1.legend(loc='best')
        plt.title('Training Loss')

    plt.tight_layout()  # 确保布局合理
    plt.show()


def conf_matrix(conf_matrix):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.ylabel('label')
    plt.xlabel('pred')
    plt.title('confusion_matrix')
    plt.show()


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


def plot_ts(
    df,  # 包含时间戳和多特征的DataFrame
    feature_cols=None,  # 要绘制的特征列列表，None则自动取非时间戳列
    max_samples=None  # 时间轴长度
):
    """从DataFrame绘制多维时间序列曲线"""
    # 验证索引是否为时间戳格式
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        # 尝试将索引转换为时间戳
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"索引无法转换为时间戳格式: {str(e)}")
        
    # 只取前max_samples个样本（核心修改）
    if max_samples:
        total_samples = len(df)
        if total_samples > max_samples:
            df = df.iloc[:max_samples].copy()  # 取前max_samples行
            print(f"Plotting first {max_samples} samples (total: {total_samples})")
        else:
            df = df.copy()  # 数据量不足时全取
            print(f"Plotting all {total_samples} samples (less than max_samples)")

    # 确定要绘制的特征列
    if feature_cols is None:
        feature_cols = df.columns.tolist()  # 默认为所有列
    else:
        # 检查指定的特征列是否存在
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame中不存在这些列: {missing_cols}")

    num_features = len(feature_cols)
    if num_features == 0:
        raise ValueError("没有可绘制的特征列")

    # 创建画布和轴
    fig, ax = plt.subplots(figsize=(12, 6))

    # 获取区分度高的颜色
    colors = list(get_named_colors_mapping().values())
    step = max(1, int(len(colors) / num_features))  # 避免颜色重复
    selected_colors = colors[::step][:num_features]

    # 绘制每条曲线（使用索引作为时间轴）
    for i, feature in enumerate(feature_cols):
        ax.plot(
            df.index,  # 使用索引（时间戳）作为x轴
            df[feature],
            label=feature,
            color=selected_colors[i],
            linewidth=1.5,
            alpha=0.8
        )

    # 设置时间轴格式
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(r"%Y-%m-%d %H:%M"))  # 年-月-日 时:分
    plt.xticks(rotation=45)  # 旋转标签避免重叠
    fig.autofmt_xdate()  # 自动调整日期显示格式

    # 添加标题和标签
    ax.set_title("Multivariate Time Series Trend (Timestamp as Index)", fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)

    # 添加网格和图例
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        loc="best",
        fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="gray"
    )

    # 调整布局
    plt.tight_layout()

    plt.show()
