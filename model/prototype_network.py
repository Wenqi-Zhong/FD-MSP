import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class MultiScalePrototypeLayer(nn.Module):
    """多尺度原型层，基于ScaleProtoSeg的实现，只为息肉类别设置原型"""
    
    def __init__(self, feature_dim: int, prototype_shape: Tuple[int, int, int, int], 
                 num_classes: int, num_scales: int = 4, activation_function: str = 'log'):
        super(MultiScalePrototypeLayer, self).__init__()
        
        self.feature_dim = feature_dim
        self.prototype_shape = prototype_shape
        self.num_classes = num_classes
        self.num_scales = num_scales
        self.activation_function = activation_function
        self.epsilon = 1e-4
        
        # 只为息肉类别设置原型（类别1），所有原型都属于息肉类别
        self.prototype_vectors = nn.Parameter(torch.rand(prototype_shape), requires_grad=True)
        
        # 用于计算距离的权重
        self.ones = nn.Parameter(torch.ones(prototype_shape), requires_grad=False)
        
        # 原型类别身份矩阵 - 所有原型都属于息肉类别（类别1）
        self.prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        # 所有原型都分配给息肉类别（类别1）
        self.prototype_class_identity[:, 1] = 1  # 息肉类别
        
        # 特征适配层
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(feature_dim, prototype_shape[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(prototype_shape[1], prototype_shape[1], kernel_size=1),
            nn.Sigmoid()
        )
        
        # 多尺度特征提取
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv2d(prototype_shape[1], prototype_shape[1], kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(prototype_shape[1], prototype_shape[1], kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(prototype_shape[1], prototype_shape[1], kernel_size=3, padding=4, dilation=4),
            nn.Conv2d(prototype_shape[1], prototype_shape[1], kernel_size=3, padding=8, dilation=8)
        ])
        
        self._initialize_weights()
    
    @property
    def num_prototypes(self) -> int:
        return self.prototype_vectors.shape[0]
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.feature_adapter.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.multi_scale_convs:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _l2_convolution(self, x: torch.Tensor, prototype_vectors: torch.Tensor, 
                       ones: torch.Tensor) -> torch.Tensor:
        """计算L2距离卷积"""
        # x: [batch, channels, h, w]
        # prototype_vectors: [num_prototypes, channels, 1, 1]
        batch_size, channels, h, w = x.shape
        num_prototypes = prototype_vectors.shape[0]
        
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=ones)  # [batch, num_prototypes, h, w]
        
        p2 = prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))  # [num_prototypes]
        p2_reshape = p2.view(1, -1, 1, 1)  # [1, num_prototypes, 1, 1]
        
        xp = F.conv2d(input=x, weight=prototype_vectors)  # [batch, num_prototypes, h, w]
        intermediate_result = -2 * xp + p2_reshape
        distances = F.relu(x2_patch_sum + intermediate_result)
        
        return distances
    
    def _multi_scale_l2_convolution(self, x: torch.Tensor) -> torch.Tensor:
        """多尺度L2距离计算 - 增强版本，加权融合不同尺度"""
        # 使用不同膨胀率的卷积来模拟多尺度
        multi_scale_features = []
        scale_weights = []
        
        for i, conv in enumerate(self.multi_scale_convs):
            scale_feat = conv(x)
            multi_scale_features.append(scale_feat)
        
            # 动态学习尺度权重，小尺度更关注边界，大尺度更关注整体形状
            scale_weight = torch.mean(scale_feat, dim=[2, 3], keepdim=True)
            scale_weights.append(scale_weight)
        
        # 归一化权重
        scale_weights = torch.stack(scale_weights, dim=0)  # [num_scales, batch, channels, 1, 1]
        scale_weights = F.softmax(scale_weights, dim=0)
        
        # 加权融合多尺度特征
        fused_features = torch.zeros_like(multi_scale_features[0])
        for i, feat in enumerate(multi_scale_features):
            fused_features += feat * scale_weights[i]
        
        # 添加残差连接以保持原始特征
        fused_features = fused_features + x
        
        # 计算距离
        distances = self._l2_convolution(fused_features, self.prototype_vectors, self.ones)
        
        return distances
    
    def distance_2_similarity(self, distances: torch.Tensor) -> torch.Tensor:
        """将距离转换为相似度"""
        if self.activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.activation_function == 'linear':
            return -distances
        else:
            return self.activation_function(distances)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 特征适配
        conv_features = self.feature_adapter(x)
        
        # 计算多尺度距离
        distances = self._multi_scale_l2_convolution(conv_features)
        
        # 计算激活
        activations = self.distance_2_similarity(distances)
        
        return distances, activations


class WeightedAggregation(nn.Module):
    """加权聚合模块，用于跨尺度信息传播"""
    
    def __init__(self, channel_dim: int, output_type: str = 'weighted'):
        super(WeightedAggregation, self).__init__()
        self.output_type = output_type
        self.channel_dim = channel_dim
        
        if output_type == 'weighted':
            self.weight_conv = nn.Conv2d(channel_dim, channel_dim, kernel_size=1)
            self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor, activations: torch.Tensor, 
                prototypes: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.output_type == 'weighted':
            weights = self.sigmoid(self.weight_conv(x))
            # 使用原型激活来调制特征
            prototype_influence = torch.mean(activations, dim=1, keepdim=True)
            prototype_influence = F.interpolate(prototype_influence, size=x.shape[-2:], 
                                              mode='bilinear', align_corners=True)
            return x * weights * prototype_influence
        else:
            return x


class PrototypeNetwork(nn.Module):
    """完整的原型网络，整合多尺度原型学习"""
    
    def __init__(self, feature_dim: int, prototype_shape: Tuple[int, int, int, int],
                 num_classes: int, num_scales: int = 4, 
                 activation_function: str = 'log', scale_head_type: Optional[str] = None):
        super(PrototypeNetwork, self).__init__()
        
        self.feature_dim = feature_dim
        self.prototype_shape = prototype_shape
        self.num_classes = num_classes
        self.num_scales = num_scales
        
        # 多尺度原型层
        self.prototype_layer = MultiScalePrototypeLayer(
            feature_dim=feature_dim,
            prototype_shape=prototype_shape,
            num_classes=num_classes,
            num_scales=num_scales,
            activation_function=activation_function
        )
        
        # 尺度聚合头
        if scale_head_type is not None:
            self.scale_head = WeightedAggregation(
                channel_dim=prototype_shape[1],
                output_type=scale_head_type
            )
        else:
            self.scale_head = None
        
        # 不再需要线性分类层，因为我们使用自定义的分类逻辑
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        distances, activations = self.prototype_layer(x)
        return distances, activations
    
    def classify(self, activations: torch.Tensor) -> torch.Tensor:
        """基于原型激活进行分类，改进的息肉检测策略"""
        # 展平激活以进行分类
        batch_size, num_prototypes, h, w = activations.shape
        activations_flat = activations.permute(0, 2, 3, 1).contiguous()
        activations_flat = activations_flat.view(-1, num_prototypes)
        
        # 使用top-k原型投票机制，增强息肉检测
        k = min(5, num_prototypes)  # 使用top-5原型
        top_k_values, top_k_indices = torch.topk(activations_flat, k, dim=1)
        
        # 加权平均top-k激活，距离越近权重越大
        weights = F.softmax(top_k_values, dim=1)
        polyp_activation = torch.sum(top_k_values * weights, dim=1, keepdim=True)
        
        # 添加置信度调制：基于激活的方差来判断可靠性
        activation_var = torch.var(activations_flat, dim=1, keepdim=True)
        confidence = torch.sigmoid(-activation_var + 0.5)  # 方差小时置信度高
        
        # 背景和息肉的概率计算
        polyp_prob = torch.sigmoid(polyp_activation) * confidence
        background_prob = 1.0 - polyp_prob
        
        # 增强边界区域的息肉检测
        # 对于激活值中等的区域，给予更多息肉倾向
        medium_activation_mask = (polyp_activation > 0.3) & (polyp_activation < 0.7)
        polyp_prob = torch.where(medium_activation_mask, 
                                polyp_prob * 1.2,  # 提升中等激活区域的息肉概率
                                polyp_prob)
        
        # 确保概率和为1
        total_prob = background_prob + polyp_prob
        background_prob = background_prob / total_prob
        polyp_prob = polyp_prob / total_prob
        
        # 组合logits (转换为logit空间)
        background_logit = torch.log(background_prob + 1e-8)
        polyp_logit = torch.log(polyp_prob + 1e-8)
        logits = torch.cat([background_logit, polyp_logit], dim=1)
        
        # 重塑回原始形状
        logits = logits.view(batch_size, h, w, self.num_classes)
        logits = logits.permute(0, 3, 1, 2).contiguous()
        
        return logits
    
    def get_prototype_class_identity(self):
        """获取原型类别身份矩阵"""
        return self.prototype_layer.prototype_class_identity


class PrototypeClassifier(nn.Module):
    """原型分类器，用于将原型激活转换为类别预测"""
    
    def __init__(self, num_prototypes: int, num_classes: int):
        super(PrototypeClassifier, self).__init__()
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        
        # 分类层
        self.classifier = nn.Linear(num_prototypes, num_classes, bias=False)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 全局平均池化
        pooled_activations = self.global_avg_pool(activations)
        pooled_activations = pooled_activations.view(pooled_activations.size(0), -1)
        
        # 分类
        logits = self.classifier(pooled_activations)
        
        return logits 