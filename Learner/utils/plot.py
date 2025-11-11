import datetime
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
    if has_dev_metrics and all_epochs:  # 确保有数据才进行排序
        # 对数据按epoch排序
        sorted_data = sorted(zip(all_epochs, all_train_losses,
                                 all_dev_losses, all_dev_scores), key=lambda x: x[0])
        epochs, train_losses, dev_losses, dev_scores = zip(
            *sorted_data)

    elif all_epochs:  # 仅有训练数据
        sorted_data = sorted(
            zip(all_epochs, all_train_losses), key=lambda x: x[0])
        epochs, train_losses = zip(*sorted_data)

    else:
        print("未找到有效的训练数据")
        return

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

        # 绘制分数曲线（右Y轴）- 优化缩放以占满图表
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Score', color=color)
        ax2.plot(epochs, dev_scores,
                 color=color, label='Validation Score')
        ax2.tick_params(axis='y', labelcolor=color)

        # 计算分数数据范围
        score_range = max(dev_scores) - min(dev_scores)

        # 动态调整分数轴范围，确保曲线占满图表
        # 根据数据分布自动调整边距，数据范围越小，相对边距越大
        if score_range < 0.1:  # 数据范围很小时，使用较大边距
            margin = 0.05
        elif score_range < 0.5:  # 数据范围中等时，使用中等边距
            margin = score_range * 0.2
        else:  # 数据范围较大时，使用较小边距
            margin = score_range * 0.1

        # 设置坐标轴范围，确保数据占满图表
        ax2.set_ylim(
            min(dev_scores) - margin,
            max(dev_scores) + margin
        )

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
    total_steps=50,  # 总训练轮数
    label="Learning Rate Schedule"  # 曲线标签（用于多曲线对比）
):
    """绘制学习率调度器的学习率变化曲线"""
    # 绘制学习率曲线
    lrs = []
    for epoch in range(total_steps):
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        # print(f"Epoch {epoch+1}: LR = {current_lr:.6f}")
        scheduler.step()  # 每个epoch更新一次学习率

    # 可视化（可选）
    plt.plot(range(1, total_steps+1), lrs, label=label, linewidth=2)
    plt.xlabel("STep")
    plt.ylabel("Learning Rate")
    # plt.title("SequentialLR: Warmup + Cosine Annealing")
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
            print(
                f"Plotting first {max_samples} samples (total: {total_samples})")
        else:
            df = df.copy()  # 数据量不足时全取
            print(
                f"Plotting all {total_samples} samples (less than max_samples)")

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
    ax.set_title(
        "Multivariate Time Series Trend (Timestamp as Index)", fontsize=14)
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


def plot_forecast_comparison(
    original_series,  # numpy数组，原始时间序列数据
    predicted_series,  # numpy数组，模型预测的时间序列数据
    start_date=None,  # 可选， datetime对象，序列的起始日期
    time_interval='day',  # 时间间隔，可选值：'day', 'hour', 'minute'
    title='Comparison of time series forecasting',  # 图表标题
    xlabel='Time',  # x轴标签
    ylabel='Value',  # y轴标签
    figsize=(12, 6)  # 图表大小
):
    """绘制时间序列预测的原始序列与预测序列对比图"""
    # 确保两个序列长度一致
    if len(original_series) != len(predicted_series):
        raise ValueError("原始序列和预测序列的长度必须一致")

    # 生成时间轴
    n_points = len(original_series)
    if start_date is None:
        # 如果没有提供起始日期，使用索引作为时间轴
        time_axis = np.arange(n_points)
        use_dates = False
    else:
        # 根据起始日期和时间间隔生成日期时间轴
        use_dates = True
        if time_interval == 'day':
            time_axis = [start_date +
                         datetime.timedelta(days=i) for i in range(n_points)]
        elif time_interval == 'hour':
            time_axis = [start_date +
                         datetime.timedelta(hours=i) for i in range(n_points)]
        elif time_interval == 'minute':
            time_axis = [start_date +
                         datetime.timedelta(minutes=i) for i in range(n_points)]
        else:
            raise ValueError("time_interval必须是'day', 'hour'或'minute'")

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制原始序列
    ax.plot(time_axis, original_series, label='Original series',
            color='tab:blue', linewidth=2, linestyle='-')

    # 绘制预测序列
    ax.plot(time_axis, predicted_series, label='Predicted series',
            color='tab:orange', linewidth=2, linestyle='-')

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 设置标签和标题
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)

    # 设置x轴日期格式（如果使用日期）
    if use_dates:
        ax.xaxis.set_major_locator(mdates.DayLocator(
            interval=max(1, n_points//10)))  # 自动调整刻度密度
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')  # 旋转日期标签，避免重叠

    # 添加图例
    ax.legend(fontsize=10, loc='best')

    # 调整布局
    plt.tight_layout()

    plt.show()
