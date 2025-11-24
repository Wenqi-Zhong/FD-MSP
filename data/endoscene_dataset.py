import os
import torch
import numpy as np
import scipy.misc as m
from tqdm import tqdm

from torch.utils import data
from PIL import Image

from data.augmentations import *
from data.base_dataset import BaseDataset
from data.randaugment import RandAugmentMC

import random
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

class EndoScene_loader(BaseDataset):
    """EndoScene数据集加载器
    
    用于加载息肉检测和分割的数据
    """

    def __init__(self, opt, logger, augmentations = None, split='train'):
        """__init__

        :param opt: parameters of dataset
        :param logger: logging file
        :param augmentations: 
        """
        
        self.opt = opt
        self.root = opt.tgt_rootpath
        self.split = split
        self.augmentations = augmentations
        self.randaug = RandAugmentMC(2, 10)
        self.n_classes = 2  # 息肉分割是二分类任务：背景和息肉
        self.mean = np.array([0.485, 0.456, 0.406])  # ImageNet标准化均值
        self.std = np.array([0.229, 0.224, 0.225])   # ImageNet标准化标准差
        self.files = {}
        self.paired_files = {}

        if logger is not None:
            logger.info("pseudo_labels_folder set to {}".format(getattr(opt, "pseudo_labels_folder", "None")))

        self.images_base = os.path.join(self.root, "images")
        
        # 根据是否指定了pseudo_labels_folder来决定使用原始mask还是伪标签
        if hasattr(opt, "pseudo_labels_folder") and opt.pseudo_labels_folder:
            self.annotations_base = os.path.join(opt.pseudo_labels_folder)
        else:
            self.annotations_base = os.path.join(self.root, "masks")

        self.files = sorted(recursive_glob(rootdir=self.images_base, suffix=".png"))
        if not self.files:
            self.files = sorted(recursive_glob(rootdir=self.images_base, suffix=".jpg"))

        if not self.files:
            raise Exception(
                "No files for split=[%s] found in %s" % (self.split, self.images_base)
            )

        self.valid_classes = [0, 1]  # 0: 背景, 1: 息肉
        self.class_names = ["background", "polyp"]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

        clrjit_params = getattr(opt, "clrjit_params", [0.5, 0.5, 0.5, 0.2])
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(*clrjit_params),
        ])

        print("Found %d %s images" % (len(self.files), self.split))
    
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
            # 如果没有找到对应的mask，创建一个全黑的mask（假设没有息肉）
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
        
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)

        img_full = img.copy().astype(np.float64)
        img_full = (img_full / 255.0 - self.mean) / self.std
        img_full = img_full.transpose(2, 0, 1)
        lbl_full = lbl.copy()

        lp, lpsoft, weak_params = None, None, None
        
        # 加载软标签（如果存在）
        if self.split == 'train' and hasattr(self.opt, "soft_labels_folder") and self.opt.soft_labels_folder:
            lpsoft_path = os.path.join(self.opt.soft_labels_folder, os.path.basename(img_path).replace('.png', '.npy'))
            if os.path.exists(lpsoft_path):
                lpsoft = np.load(lpsoft_path)
                # 确保lpsoft的形状是 [2, H, W]（二分类任务）
                if lpsoft.ndim == 2:  # 如果是 [H, W]
                    lpsoft = np.expand_dims(lpsoft, axis=0)  # 变成 [1, H, W]
                
                if lpsoft.shape[0] == 1 and self.n_classes > 1:
                    # 创建一个全零的2通道数组
                    new_lpsoft = np.zeros((self.n_classes, lpsoft.shape[1], lpsoft.shape[2]), dtype=lpsoft.dtype)
                    # 将原始单通道数据的值作为索引，在对应位置设置为1
                    for h in range(lpsoft.shape[1]):
                        for w in range(lpsoft.shape[2]):
                            idx = int(lpsoft[0, h, w])
                            if idx < self.n_classes:
                                new_lpsoft[idx, h, w] = 1.0
                    lpsoft = new_lpsoft
            else:
                # 如果文件不存在，创建一个全零的2通道数组
                lpsoft = np.zeros((self.n_classes, img_size[1], img_size[0]), dtype=np.float32)
        
        input_dict = {}
        if self.augmentations!=None:
            img, lbl, lp, lpsoft, weak_params = self.augmentations(img, lbl, lp, lpsoft)
            img_strong, params = self.randaug(Image.fromarray(img))
            img_strong, _, _ = self.transform(img_strong, lbl)
            input_dict['img_strong'] = img_strong
            input_dict['params'] = params

        img = self.train_transform(img)

        img, lbl_, lp = self.transform(img, lbl, lp)
                
        input_dict['img'] = img
        input_dict['img_full'] = torch.from_numpy(img_full).float()
        input_dict['label'] = lbl_
        input_dict['lp'] = lp
        input_dict['lpsoft'] = lpsoft
        input_dict['weak_params'] = weak_params
        input_dict['img_path'] = self.files[index]
        input_dict['lbl_full'] = torch.from_numpy(lbl_full).long()

        input_dict = {k:v for k, v in input_dict.items() if v is not None}
        return input_dict

    def transform(self, img, lbl, lp=None, check=True):
        """transform

        :param img:
        :param lbl:
        """
        img = np.array(img)
        img = img.astype(np.float64)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        lbl = lbl.astype(int)

        if check and not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if lp is not None:
            classes = np.unique(lp)
            lp = np.array(lp)
            lp = torch.from_numpy(lp).long()

        return img, lbl, lp

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
    
    def encode_segmap(self, mask):
        """
        将任何掩码编码为标准分割图
        """
        # 二值化
        mask = (mask > 0).astype(np.uint8)
        return mask

    def get_cls_num_list(self):
        """
        获取类别数量统计
        """
        cls_num_list = np.array([0, 0])  # 初始化为背景和息肉两个类别
        
        for index in tqdm(range(len(self.files))):
            img_path = self.files[index].rstrip()
            file_name = os.path.basename(img_path)
            lbl_path = os.path.join(self.annotations_base, file_name)
            
            if os.path.exists(lbl_path):
                lbl = Image.open(lbl_path).convert('L')
                lbl = np.array(lbl)
                lbl = (lbl > 127).astype(np.uint8)
                
                # 统计每个类别的像素数量
                cls_num_list[0] += np.sum(lbl == 0)  # 背景
                cls_num_list[1] += np.sum(lbl == 1)  # 息肉
        
        return cls_num_list 