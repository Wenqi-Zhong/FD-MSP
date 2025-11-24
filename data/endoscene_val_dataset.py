import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image

from data.base_dataset import BaseDataset
from torchvision import transforms

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class EndoScene_val_loader(BaseDataset):
    """EndoScene验证数据集加载器
    
    用于加载验证集的息肉图像和掩码
    """

    def __init__(self, opt, logger, augmentations = None, split='val'):
        """__init__

        :param opt: parameters of dataset
        :param logger: logging file
        :param augmentations: 
        """
        
        self.opt = opt
        # 对于验证集，我们使用ValidationDataset文件夹
        self.root = os.path.join(os.path.dirname(opt.tgt_rootpath), "ValidationDataset")
        self.augmentations = augmentations
        self.n_classes = 2  # 息肉分割是二分类任务：背景和息肉
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.files = {}

        # 验证集位置
        self.images_base = os.path.join(self.root, "images")
        self.annotations_base = os.path.join(self.root, "masks")

        self.files = sorted(recursive_glob(rootdir=self.images_base, suffix=".png"))
        if not self.files:
            self.files = sorted(recursive_glob(rootdir=self.images_base, suffix=".jpg"))

        if not self.files:
            raise Exception("No files found in %s" % (self.images_base))

        self.valid_classes = [0, 1]  # 0: 背景, 1: 息肉
        self.class_names = ["background", "polyp"]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

        if logger is not None:
            logger.info("Found %d %s images" % (len(self.files), split))
        print("Found %d %s images" % (len(self.files), split))
    
    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[index].rstrip()
        # 计算对应的mask路径
        file_name = os.path.basename(img_path)
        lbl_path = os.path.join(self.annotations_base, file_name)
        
        # 如果mask文件不存在，尝试不同的扩展名
        if not os.path.exists(lbl_path):
            name_without_ext = os.path.splitext(file_name)[0]
            possible_exts = ['.png', '.jpg', '.jpeg']
            for ext in possible_exts:
                test_path = os.path.join(self.annotations_base, name_without_ext + ext)
                if os.path.exists(test_path):
                    lbl_path = test_path
                    break

        img = Image.open(img_path).convert('RGB')
        
        if os.path.exists(lbl_path):
            lbl = Image.open(lbl_path).convert('L')
        else:
            lbl = Image.new('L', img.size, 0)
            
        # 将二值掩码转换为二分类掩码（0表示背景，1表示息肉）
        lbl = np.array(lbl)
        lbl = (lbl > 127).astype(np.uint8)  # 阈值化为二值图像
        lbl = Image.fromarray(lbl)
        
        # 调整图像大小
        if hasattr(self.opt, 'resize'):
            img_size = (self.opt.resize, self.opt.resize)
        else:
            img_size = (384, 384)  # 默认大小
            
        img = img.resize(img_size, Image.BILINEAR)
        lbl = lbl.resize(img_size, Image.NEAREST)
        
        # 执行数据增强（如果有）
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        # 转换为张量
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)

        # 标准化和转置
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose(2, 0, 1)

        # 转换为PyTorch张量
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        input_dict = {}
        input_dict['img'] = img
        input_dict['label'] = lbl
        input_dict['img_path'] = img_path
        input_dict['lbl_path'] = lbl_path
        
        return input_dict
        
    def decode_segmap(self, temp):
        """
        解码分割图为彩色图像
        """
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        
        # 背景 - 黑色
        r[temp == 0] = 0
        g[temp == 0] = 0
        b[temp == 0] = 0
        
        # 息肉 - 红色
        r[temp == 1] = 255
        g[temp == 1] = 0
        b[temp == 1] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        
        return rgb 