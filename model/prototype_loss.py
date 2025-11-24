import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class PrototypeLoss(nn.Module):
    """原型损失函数，专门针对只有息肉原型的情况"""
    
    def __init__(self, num_classes: int, num_prototypes: int, 
                 lambda_diversity: float = 0.01, lambda_separation: float = 0.01):
        super(PrototypeLoss, self).__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.lambda_diversity = lambda_diversity
        self.lambda_separation = lambda_separation
    
    def diversity_loss(self, prototype_vectors: torch.Tensor) -> torch.Tensor:
        """计算息肉原型之间的多样性损失"""
        # 重塑原型向量
        prototypes = prototype_vectors.view(self.num_prototypes, -1)
        
        # 计算原型之间的余弦相似度
        normalized_prototypes = F.normalize(prototypes, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_prototypes, normalized_prototypes.t())
        
        # 移除对角线元素（自相似度）
        mask = torch.eye(self.num_prototypes, device=prototype_vectors.device)
        similarity_matrix = similarity_matrix * (1 - mask)
        
        # 多样性损失：最小化息肉原型之间的相似度
        diversity_loss = torch.mean(torch.abs(similarity_matrix))
        
        return diversity_loss
    
    def cluster_loss(self, distances: torch.Tensor, activations: torch.Tensor, 
                    labels: torch.Tensor) -> torch.Tensor:
        """计算聚类损失，改进的息肉像素聚类策略"""
        batch_size, num_prototypes, h, w = distances.shape
        
        # 重塑标签和距离
        labels_flat = labels.view(-1)  # [batch*h*w]
        distances_flat = distances.permute(0, 2, 3, 1).contiguous().view(-1, num_prototypes)  # [batch*h*w, num_prototypes]
        activations_flat = activations.permute(0, 2, 3, 1).contiguous().view(-1, num_prototypes)  # [batch*h*w, num_prototypes]
        
        # 确保维度匹配
        assert labels_flat.shape[0] == distances_flat.shape[0], f"Labels shape {labels_flat.shape} doesn't match distances shape {distances_flat.shape}"
        
        # 只关注息肉像素（标签为1）
        polyp_mask = labels_flat == 1
        if torch.sum(polyp_mask) == 0:
            return torch.tensor(0.0, device=distances.device)
        
        polyp_distances = distances_flat[polyp_mask]  # [num_polyp_pixels, num_prototypes]
        polyp_activations = activations_flat[polyp_mask]  # [num_polyp_pixels, num_prototypes]
        
        # 1. 基础聚类损失：息肉像素到最近原型的距离
        min_distances, min_indices = torch.min(polyp_distances, dim=1)
        base_cluster_loss = torch.mean(min_distances)
        
        # 2. 难样本挖掘：对距离较远的息肉像素加权
        # 找出距离最远的30%作为难样本
        hard_sample_ratio = 0.3
        num_hard_samples = max(1, int(len(min_distances) * hard_sample_ratio))
        hard_distances, hard_indices = torch.topk(min_distances, num_hard_samples)
        hard_sample_loss = torch.mean(hard_distances) * 2.0  # 给难样本更大权重
        
        # 3. 多样性促进损失：鼓励不同息肉像素匹配不同原型
        # 计算原型使用分布的熵
        prototype_usage = torch.bincount(min_indices, minlength=num_prototypes).float()
        prototype_usage = prototype_usage / torch.sum(prototype_usage)
        prototype_entropy = -torch.sum(prototype_usage * torch.log(prototype_usage + 1e-8))
        max_entropy = torch.log(torch.tensor(num_prototypes, dtype=torch.float32))
        diversity_loss = (max_entropy - prototype_entropy) * 0.1
        
        # 4. 置信度加权损失：激活值高的像素给予更大权重
        activation_weights = torch.max(polyp_activations, dim=1)[0]
        activation_weights = F.softmax(activation_weights, dim=0)
        weighted_cluster_loss = torch.sum(min_distances * activation_weights)
        
        # 组合所有损失
        total_cluster_loss = (base_cluster_loss + 
                            hard_sample_loss + 
                            diversity_loss + 
                            weighted_cluster_loss * 0.5)
        
        return total_cluster_loss
    
    def forward(self, distances: torch.Tensor, activations: torch.Tensor, 
                labels: torch.Tensor, prototype_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算总的原型损失"""
        total_loss = torch.tensor(0.0, device=distances.device)
        
        # 聚类损失：鼓励息肉像素与息肉原型接近
        cluster_loss = self.cluster_loss(distances, activations, labels)
        total_loss += cluster_loss
        
        # 多样性损失：鼓励息肉原型之间的多样性
        if prototype_vectors is not None and self.lambda_diversity > 0:
            div_loss = self.diversity_loss(prototype_vectors)
            total_loss += self.lambda_diversity * div_loss
        
        # 不再需要分离损失，因为只有一个类别的原型
        
        return total_loss


class PixelWiseCrossEntropyLoss(nn.Module):
    """像素级交叉熵损失，支持忽略标签"""
    
    def __init__(self, ignore_index: Optional[int] = None, return_correct: bool = False):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.return_correct = return_correct
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    
    def forward(self, predicted_logits: torch.Tensor, target_labels: torch.Tensor):
        """计算像素级交叉熵损失"""
        loss = self.cross_entropy(predicted_logits, target_labels)
        
        if self.return_correct:
            # 计算正确预测的像素数
            with torch.no_grad():
                predicted_classes = torch.argmax(predicted_logits, dim=1)
                if self.ignore_index is not None:
                    valid_mask = target_labels != self.ignore_index
                    correct = (predicted_classes == target_labels) & valid_mask
                else:
                    correct = predicted_classes == target_labels
            return loss, correct
        else:
            return loss


class KLDivergenceLoss(nn.Module):
    """KL散度损失，用于原型激活的正则化"""
    
    def __init__(self, prototype_class_identity: torch.Tensor, num_scales: int,
                 scale_num_prototypes: dict):
        super(KLDivergenceLoss, self).__init__()
        self.prototype_class_identity = prototype_class_identity
        self.num_scales = num_scales
        self.scale_num_prototypes = scale_num_prototypes
    
    def forward(self, prototype_distances: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """计算KL散度损失"""
        batch_size, num_prototypes, h, w = prototype_distances.shape
        
        # 将距离转换为概率分布
        prototype_activations = F.softmax(-prototype_distances, dim=1)
        
        # 重塑标签
        target_labels_flat = target_labels.view(-1)
        prototype_activations_flat = prototype_activations.view(num_prototypes, -1).t()
        
        # 过滤有效像素
        valid_mask = target_labels_flat != 250  # 忽略标签
        if torch.sum(valid_mask) == 0:
            return torch.tensor(0.0, device=prototype_distances.device)
        
        valid_labels = target_labels_flat[valid_mask]
        valid_activations = prototype_activations_flat[valid_mask]
        
        kld_loss = 0.0
        count = 0
        
        # 对每个类别计算KL散度
        for c in range(self.prototype_class_identity.shape[1]):
            class_mask = valid_labels == c
            if torch.sum(class_mask) == 0:
                continue
            
            # 获取属于类别c的原型
            prototype_mask = self.prototype_class_identity[:, c] == 1
            if torch.sum(prototype_mask) == 0:
                continue
            
            # 计算目标分布（均匀分布在类别c的原型上）
            target_dist = torch.zeros(num_prototypes, device=prototype_distances.device)
            target_dist[prototype_mask] = 1.0 / torch.sum(prototype_mask.float())
            
            # 计算KL散度
            class_activations = valid_activations[class_mask]
            for activation in class_activations:
                kld = F.kl_div(torch.log(activation + 1e-8), target_dist, reduction='sum')
                kld_loss += kld
                count += 1
        
        if count > 0:
            kld_loss = kld_loss / count
        else:
            kld_loss = torch.tensor(0.0, device=prototype_distances.device)
        
        return kld_loss


class EntropySamplingLoss(nn.Module):
    """熵采样损失，鼓励原型激活的多样性"""
    
    def __init__(self, prototype_class_identity: torch.Tensor, num_scales: int,
                 scale_num_prototypes: dict):
        super(EntropySamplingLoss, self).__init__()
        self.prototype_class_identity = prototype_class_identity
        self.num_scales = num_scales
        self.scale_num_prototypes = scale_num_prototypes
    
    def forward(self, prototype_activations: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """计算熵采样损失"""
        batch_size, num_prototypes, num_patches = prototype_activations.shape
        
        # 计算激活的熵
        activations_prob = F.softmax(prototype_activations, dim=1)
        entropy = -torch.sum(activations_prob * torch.log(activations_prob + 1e-8), dim=1)
        
        # 鼓励高熵（多样性）
        entropy_loss = -torch.mean(entropy)
        
        return entropy_loss


class NormalizationLoss(nn.Module):
    """归一化损失，正则化原型激活"""
    
    def __init__(self, prototype_class_identity: torch.Tensor, norm_type: str = 'l1'):
        super(NormalizationLoss, self).__init__()
        self.prototype_class_identity = prototype_class_identity
        self.norm_type = norm_type
    
    def forward(self, prototype_activations: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """计算归一化损失"""
        if self.norm_type == 'l1':
            norm_loss = torch.mean(torch.abs(prototype_activations))
        elif self.norm_type == 'l2':
            norm_loss = torch.mean(prototype_activations ** 2)
        else:
            norm_loss = torch.tensor(0.0, device=prototype_activations.device)
        
        return norm_loss 


class BoundaryEnhancedLoss(nn.Module):
    """边界增强损失，专门提升息肉边界分割精度"""
    
    def __init__(self, boundary_weight: float = 1.0, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super(BoundaryEnhancedLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def get_boundary_mask(self, labels: torch.Tensor, dilation_radius: int = 2) -> torch.Tensor:
        """提取息肉边界区域的掩码"""
        # 使用形态学操作找到边界
        batch_size = labels.shape[0]
        boundary_masks = []
        
        for b in range(batch_size):
            label = labels[b].float()
            
            # 膨胀和腐蚀操作
            kernel = torch.ones(1, 1, 2*dilation_radius+1, 2*dilation_radius+1).to(label.device)
            padding = dilation_radius
            
            # 膨胀
            dilated = F.conv2d(label.unsqueeze(0).unsqueeze(0), kernel, padding=padding)
            dilated = (dilated > 0).float()
            
            # 腐蚀
            eroded = -F.conv2d(-label.unsqueeze(0).unsqueeze(0), kernel, padding=padding)
            eroded = (eroded > kernel.numel() - 1).float()
            
            # 边界 = 膨胀 - 腐蚀
            boundary = dilated - eroded
            boundary_masks.append(boundary.squeeze())
            
        return torch.stack(boundary_masks, dim=0)
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def boundary_dice_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                          boundary_mask: torch.Tensor) -> torch.Tensor:
        """边界区域的Dice损失"""
        # 将预测转换为概率
        pred_prob = F.softmax(pred, dim=1)[:, 1]  # 息肉类别概率
        target_float = target.float()
        
        # 只计算边界区域的Dice
        boundary_pred = pred_prob * boundary_mask
        boundary_target = target_float * boundary_mask
        
        intersection = torch.sum(boundary_pred * boundary_target, dim=[1, 2])
        union = torch.sum(boundary_pred, dim=[1, 2]) + torch.sum(boundary_target, dim=[1, 2])
        
        dice = (2.0 * intersection) / (union + 1e-8)
        return 1.0 - dice.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算边界增强损失"""
        # 获取边界掩码
        boundary_mask = self.get_boundary_mask(target)
        
        # 基础交叉熵损失
        ce_loss = F.cross_entropy(pred, target, ignore_index=250)
        
        # 边界区域的Focal损失
        boundary_focal_loss = self.focal_loss(pred, target)
        
        # 边界Dice损失
        boundary_dice_loss = self.boundary_dice_loss(pred, target, boundary_mask)
        
        # 组合损失
        total_loss = (ce_loss + 
                     self.boundary_weight * boundary_focal_loss + 
                     self.boundary_weight * boundary_dice_loss)
        
        return total_loss


class PolypAwareContrastiveLoss(nn.Module):
    """息肉感知对比损失，增强息肉特征的判别性"""
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super(PolypAwareContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算对比损失"""
        batch_size, channels, h, w = features.shape
        
        # 展平特征和标签
        features_flat = features.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, HW, C]
        labels_flat = labels.view(batch_size, -1)  # [B, HW]
        
        # 只处理有效像素（非ignore_label）
        valid_mask = labels_flat != 250
        
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(batch_size):
            batch_features = features_flat[b][valid_mask[b]]  # [valid_pixels, C]
            batch_labels = labels_flat[b][valid_mask[b]]  # [valid_pixels]
            
            if len(batch_features) == 0:
                continue
                
            # 分离息肉和背景特征
            polyp_mask = batch_labels == 1
            background_mask = batch_labels == 0
            
            if torch.sum(polyp_mask) == 0 or torch.sum(background_mask) == 0:
                continue
            
            polyp_features = batch_features[polyp_mask]
            background_features = batch_features[background_mask]
            
            # 计算类内紧致性（同类特征应该相似）
            polyp_center = torch.mean(polyp_features, dim=0, keepdim=True)
            background_center = torch.mean(background_features, dim=0, keepdim=True)
            
            # 息肉内部距离
            polyp_intra_dist = torch.mean(torch.norm(polyp_features - polyp_center, dim=1))
            
            # 背景内部距离
            background_intra_dist = torch.mean(torch.norm(background_features - background_center, dim=1))
            
            # 类间分离性（不同类特征应该远离）
            inter_dist = torch.norm(polyp_center - background_center)
            
            # 对比损失：最小化类内距离，最大化类间距离
            contrastive_loss = (polyp_intra_dist + background_intra_dist) - inter_dist + self.margin
            contrastive_loss = F.relu(contrastive_loss)  # 确保非负
            
            total_loss += contrastive_loss
            valid_batches += 1
        
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=features.device) 