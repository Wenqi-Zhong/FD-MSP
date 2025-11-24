#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class SegmentationMetrics:
    """分割任务评价指标计算类"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
    
    def update(self, pred, target):
        """更新混淆矩阵
        Args:
            pred: 预测结果 (batch_size, H, W)
            target: 真实标签 (batch_size, H, W)
        """
        if isinstance(pred, np.ndarray):
            pred = pred.flatten()
        else:
            pred = pred.cpu().numpy().flatten()
            
        if isinstance(target, np.ndarray):
            target = target.flatten()
        else:
            target = target.cpu().numpy().flatten()
        
        # 过滤无效像素
        mask = (target >= 0) & (target < self.num_classes)
        pred = pred[mask]
        target = target[mask]
        
        # 更新混淆矩阵
        for i in range(len(pred)):
            self.confusion_matrix[target[i], pred[i]] += 1
    
    def get_results(self):
        """计算所有评价指标"""
        # IoU计算
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)
        iou = intersection / (union + 1e-8)
        
        # Dice系数计算
        dice = 2 * intersection / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) + 1e-8)
        
        # 像素准确率
        pixel_acc = np.sum(np.diag(self.confusion_matrix)) / (np.sum(self.confusion_matrix) + 1e-8)
        
        # 类别准确率
        class_acc = intersection / (np.sum(self.confusion_matrix, axis=1) + 1e-8)
        mean_acc = np.nanmean(class_acc)
        
        return {
            'class_iou': iou.tolist(),
            'mIoU': np.nanmean(iou),
            'class_dice': dice.tolist(), 
            'mDice': np.nanmean(dice),
            'class_acc': class_acc.tolist(),
            'mean_acc': mean_acc,
            'pixel_acc': pixel_acc,
            'confusion_matrix': self.confusion_matrix.tolist()
        }
    
    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

def calculate_iou(pred, target, num_classes=2):
    """计算IoU指标
    Args:
        pred: 预测结果
        target: 真实标签
        num_classes: 类别数量
    Returns:
        iou: 每个类别的IoU
        miou: 平均IoU
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
            
        ious.append(iou)
    
    miou = np.nanmean(ious)
    return ious, miou

def calculate_dice(pred, target, num_classes=2):
    """计算Dice系数
    Args:
        pred: 预测结果
        target: 真实标签
        num_classes: 类别数量
    Returns:
        dice: 每个类别的Dice系数
        mdice: 平均Dice系数
    """
    dices = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        total = pred_cls.sum() + target_cls.sum()
        
        if total == 0:
            dice = float('nan')
        else:
            dice = 2 * intersection / total
            
        dices.append(dice)
    
    mdice = np.nanmean(dices)
    return dices, mdice

def calculate_pixel_accuracy(pred, target):
    """计算像素准确率"""
    correct = (pred == target).sum()
    total = pred.size
    return correct / total 