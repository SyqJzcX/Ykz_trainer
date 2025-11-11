from Learner.trainers.trainer import Trainer
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import numpy as np
from torch.cuda.amp import autocast, GradScaler


class TextClassifyTrainer(Trainer):
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
        super(TextClassifyTrainer, self).__init__(
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

    def forward_pass(self):
        """在训练集上个更新模型权重"""
        for batch in tqdm(self.train_dataloader, desc="模型训练"):
            # 数据移到GPU/CPU
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # 清空梯度
            self.optimizer.zero_grad()

            # 1. 混合精度前向传播（FP16）
            with autocast(enabled=self.use_fp16):
                outputs = outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            return loss, labels.size(0)  # 当前批次样本数

    def dev(self):
        """在发展集上验证模型，并更新学习率"""
        for batch in tqdm(self.dev_dataloader, desc="模型验证"):
            # 第一步：调用forward_pass，获取loss、logits、batch_size（验证模式：is_training=False）
            loss, logits, batch_size = self.forward_pass(batch, is_training=False)

            # 第二步：调用compute_dev_metrics，计算当前批次准确率
            acc = self.compute_dev_metrics(logits, batch["labels"])  # labels从原始batch取（已在forward_pass移到设备）
            
            # 第三步：逐批次yield结果（与原有接口完全兼容）
            yield loss, acc, batch_size

    def predict(self, best_model_path, test_dataloader, id2label, result_path='result.txt'):
        """
        模型预测：为测试集生成标签并保存到result.txt
        Args:
            test_dataloader: 测试集DataLoader（需用NewsTestDataset和test_collate_fn）
            id2label: 类别ID→文本标签的映射字典（{0: 'label0', 1: 'label1', ..., 13: 'label13'}）
            result_path: 预测结果保存路径
        """
        # 加载最佳模型权重
        if os.path.exists(best_model_path):
            checkpoint = torch.load(  # 加载保存的检查点
                best_model_path,
                map_location=self.device,
                weights_only=False
            )
            self.model.load_state_dict(
                checkpoint['model_state_dict'])
            print(f"已加载最佳模型权重：{best_model_path}")
        else:
            print("未找到best_model.pth，使用当前模型权重进行预测")
        
        self.model.eval()  # 切换为评估模式
        predictions = []  # 存储 (original_idx, pred_label) 元组
        
        with torch.no_grad():  # 禁用梯度计算，节省显存
            for batch in tqdm(test_dataloader, desc="模型预测"):
                # 数据移到GPU/CPU
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                original_idx = batch["original_idx"].cpu().numpy()  # 原始索引（CPU）
                
                # 模型前向传播（仅需logits，无需labels）
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits  # 预测分数 (batch_size, 14)
                
                # 转换为类别ID（取logits最大值对应的索引）
                pred_ids = torch.argmax(logits, dim=1).cpu().numpy()  # (batch_size,)
                
                # 转换为文本标签，并关联原始索引
                for idx, pred_id in zip(original_idx, pred_ids):
                    pred_label = id2label[pred_id]  # ID→文本标签
                    predictions.append( (idx, pred_label) )
        
        # 按原始索引排序（确保输出顺序与test.csv完全一致）
        predictions.sort(key=lambda x: x[0])
        # 提取排序后的标签（去掉索引）
        pred_labels = [label for _, label in predictions]
        
        # 写入result.txt（一行一个标签）
        with open(result_path, 'w', encoding='utf-8') as f:
            for label in pred_labels:
                f.write(f"{label}\n")
        
        print(f"预测完成！共处理 {len(pred_labels)} 个样本，结果已保存到：{result_path}")
        return pred_labels  # 可选：返回预测标签列表

    def eval(self, best_model_path, eval_dataloader, id2label):
        """
        评估带标签数据集的分类性能，输出核心指标
        Args:
            eval_dataloader: 带标签的评估集DataLoader（需包含 input_ids、attention_mask、labels）
            id2label: 类别ID→文本标签的映射字典（{0: '体育', 1: '财经', ...}）
            save_report_path: 分类报告保存路径（如"eval_report.txt"，None则不保存）
            plot_confusion_matrix: 是否绘制混淆矩阵（默认False）
        Returns:
            eval_metrics: 评估指标字典（整体准确率、各类别准确率等）
        """
        # 加载最佳模型权重（与predict方法一致）
        if os.path.exists(best_model_path):
            checkpoint = torch.load(  # 加载保存的检查点
                best_model_path,
                map_location=self.device,
                weights_only=False
            )
            self.model.load_state_dict(
                checkpoint['model_state_dict'])
            print(f"已加载最佳模型权重：{best_model_path}")
        else:
            print(f"未找到{best_model_path}，使用当前模型权重进行预测")
        
        self.model.eval()  # 切换为评估模式
        all_true_ids = []  # 所有样本的真实类别ID
        all_pred_ids = []  # 所有样本的预测类别ID
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="模型评估"):
                # 数据移到GPU/CPU
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                true_ids = batch["labels"].cpu().numpy()  # 真实类别ID（CPU）
                
                # 模型前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=1).cpu().numpy()  # 预测类别ID
                
                # 收集所有样本的真实ID和预测ID
                all_true_ids.extend(true_ids)
                all_pred_ids.extend(pred_ids)
        
        # 转换为numpy数组（方便计算指标）
        all_true_ids = np.array(all_true_ids)
        all_pred_ids = np.array(all_pred_ids)
        num_classes = len(id2label)
        
        # -------------------------- 计算核心指标 --------------------------
        # 1. 整体准确率
        overall_acc = accuracy_score(all_true_ids, all_pred_ids)
        
        # 2. 各类别准确率（精确率、召回率、F1-score）
        class_report = classification_report(
            all_true_ids,
            all_pred_ids,
            target_names=[id2label[idx] for idx in range(num_classes)],  # 类别名称
            labels=range(num_classes),  # 所有类别（包括样本数为0的类）
            output_dict=True,
            zero_division=0  # 样本数为0的类，指标设为0
        )
        
        # 3. 混淆矩阵（行：真实类别，列：预测类别）
        conf_matrix = confusion_matrix(all_true_ids, all_pred_ids, labels=range(num_classes))
        
        # -------------------------- 整理指标结果 --------------------------
        eval_metrics = {
            "overall_accuracy": overall_acc,
            # "class_accuracy": {id2label[idx]: class_report[idx]["precision"] for idx in range(num_classes)},
            # "class_recall": {id2label[idx]: class_report[idx]["recall"] for idx in range(num_classes)},
            # "class_f1": {id2label[idx]: class_report[idx]["f1-score"] for idx in range(num_classes)},
            "confusion_matrix": conf_matrix
        }
        
        # -------------------------- 打印评估结果 --------------------------
        print("="*50)
        print(f"评估结果汇总（共 {len(all_true_ids)} 个样本）")
        print("="*50)
        print(f"整体准确率：{overall_acc:.4f}")
        # print("\n各类别性能指标（精确率/召回率/F1-score）：")
        # # 打印表格形式的各类别指标
        # for idx in range(num_classes):
        #     label_name = id2label[idx]
        #     precision = class_report[idx]["precision"]
        #     recall = class_report[idx]["recall"]
        #     f1 = class_report[idx]["f1-score"]
        #     support = class_report[idx]["support"]  # 该类别的样本数
        #     print(f"  {label_name:>6} | 精确率：{precision:.4f} | 召回率：{recall:.4f} | F1：{f1:.4f} | 样本数：{support}")
        
        return eval_metrics
