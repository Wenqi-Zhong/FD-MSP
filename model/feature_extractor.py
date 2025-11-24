import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from .resnet import resnet101
# from torchvision.models._utils import IntermediateLayerGetter
# from mmcv.runner import load_checkpoint

# 自定义IntermediateLayerGetter
class IntermediateLayerGetter(nn.Module):
    def __init__(self, model, return_layers):
        super(IntermediateLayerGetter, self).__init__()
        self.model = model
        self.return_layers = return_layers
        
    def forward(self, x):
        out = {}
        for name, module in self.model.named_children():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        output = F.batch_norm(x, self.running_mean, self.running_var, weight=self.weight, bias=self.bias, training=False)
        return output

def resnet_feature_extractor(name, path='', freeze_bn=False, pretrained=True, aux=True):
    # name = 'resnet101'
    backbone = resnet101(pretrained=False)
    
    if pretrained and path:  # 只有当pretrained为True且path不为空时才加载权重
        if path.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(path, progress=True)
        else:
            checkpoint = torch.load(path)
            
        # 检查是否是原始模型或sourceonly模型
        if 'model_state_dict' in checkpoint:
            # 如果是训练过的模型，使用模型状态字典
            model_dict = checkpoint['model_state_dict']
            
            # 去除"module."前缀(如果有)
            new_model_dict = {}
            for k, v in model_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # 移除'module.'前缀
                if k.startswith('backbone.'):
                    k = k[9:]  # 移除'backbone.'前缀
                new_model_dict[k] = v
                
            backbone.load_state_dict(new_model_dict, strict=False)
        else:
            # 如果是PyTorch预训练的ResNet模型
            # 尝试直接加载
            try:
                backbone.load_state_dict(checkpoint, strict=False)
            except Exception as e:
                print(f"直接加载失败: {e}")
                # 检查checkpoint键结构并调整
                if 'backbone.conv1.weight' in checkpoint:
                    # 需要去除backbone.前缀
                    new_checkpoint = {}
                    for k, v in checkpoint.items():
                        if k.startswith('backbone.'):
                            new_k = k[9:]  # 'backbone.'的长度是9
                            new_checkpoint[new_k] = v
                        else:
                            new_checkpoint[k] = v
                    backbone.load_state_dict(new_checkpoint, strict=False)
                else:
                    print("无法确定模型结构，请检查预训练模型")

    if 'resnet' in name:
        backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
    else:
        NotImplementedError

    return backbone
