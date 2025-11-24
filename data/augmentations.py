# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import inspect
import cv2
from PIL import Image, ImageOps, ImageFilter


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask, mask1=None, lpsoft=None):
        params = {}  # 初始化params为空字典
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            if mask1 is not None:
                mask1 = Image.fromarray(mask1, mode="L")
            if lpsoft is not None:
                lpsoft = torch.from_numpy(lpsoft)
                if lpsoft.dim() == 1:
                    lpsoft = lpsoft.view(1, 1, -1)
                elif lpsoft.dim() == 2:
                    lpsoft = lpsoft.unsqueeze(0)
                lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[img.size[1], img.size[0]], mode='bilinear', align_corners=True)[0]
            self.PIL2Numpy = True

        if img.size != mask.size:
            print (img.size, mask.size)
        assert img.size == mask.size
        if mask1 is not None:
            assert (img.size == mask1.size)
            
        for a in self.augmentations:
            # 检查函数签名
            try:
                if len(inspect.signature(a.__call__).parameters) >= 5:  # 函数接受至少5个参数
                    img, mask, mask1, lpsoft, params = a(img, mask, mask1, lpsoft, params)
                elif len(inspect.signature(a.__call__).parameters) == 2:  # 旧函数只接受2个参数
                    img, mask = a(img, mask)
                else:
                    # 默认情况，尝试用现有参数调用
                    img, mask, mask1, lpsoft, params = a(img, mask, mask1, lpsoft, params)
            except (TypeError, AttributeError, ValueError):
                # 如果上述方法失败，尝试基本调用
                result = a(img, mask)
                if isinstance(result, tuple):
                    if len(result) == 2:
                        img, mask = result
                    elif len(result) == 5:
                        img, mask, mask1, lpsoft, params = result

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8) 
            if mask1 is not None:
                mask1 = np.array(mask1, dtype=np.uint8)
        return img, mask, mask1, lpsoft, params


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, mask1=None, lpsoft=None, params=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            if mask1 is not None:
                mask1 = ImageOps.expand(mask1, border=self.padding, fill=0)

        assert img.size == mask.size
        if mask1 is not None:
            assert (img.size == mask1.size)
        w, h = img.size

        # print("self.size: ", self.size)

        tw, th = self.size
        # if w == tw and h == th:
        #     return img, mask
        if w < tw or h < th:
            if lpsoft is not None:
                # 检查并处理lpsoft的维度
                if lpsoft.dim() == 1:
                    lpsoft = lpsoft.view(1, 1, -1)
                elif lpsoft.dim() == 2:
                    lpsoft = lpsoft.unsqueeze(0)
                lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[th, tw], mode='bilinear', align_corners=True)[0]
            if mask1 is not None:
                return (
                        img.resize((tw, th), Image.BILINEAR),
                        mask.resize((tw, th), Image.NEAREST),
                        mask1.resize((tw, th), Image.NEAREST),
                        lpsoft
                    )
            else:
                    return (
                        img.resize((tw, th), Image.BILINEAR),
                        mask.resize((tw, th), Image.NEAREST),
                        None,
                        lpsoft
                    )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        params['RandomCrop'] = (y1, y1 + th, x1, x1 + tw)
        if lpsoft is not None:
            lpsoft = lpsoft[:, y1:y1 + th, x1:x1 + tw]
        if mask1 is not None:
            return (
                img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th)),
                mask1.crop((x1, y1, x1 + tw, y1 + th)),
                lpsoft,
                params
            )
        else:
            return (
                img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th)),
                None,
                lpsoft,
                params
            )


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_saturation(img, 
                                    random.uniform(1 - self.saturation, 
                                                   1 + self.saturation)), mask


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, 
                                                  self.hue)), mask


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_brightness(img, 
                                    random.uniform(1 - self.bf, 
                                                   1 + self.bf)), mask

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_contrast(img, 
                                  random.uniform(1 - self.cf, 
                                                 1 + self.cf)), mask

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask, mask1=None, lpsoft=None, params=None):
        if random.random() < self.p:
            params['RandomHorizontallyFlip'] = True
            if lpsoft is not None:
                inv_idx = torch.arange(lpsoft.size(2)-1,-1,-1).long()  # C x H x W
                lpsoft = lpsoft.index_select(2,inv_idx)
            if mask1 is not None:
                return (
                    img.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT),
                    mask1.transpose(Image.FLIP_LEFT_RIGHT),
                    lpsoft,
                    params
                )
            else:
                return (
                    img.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT),
                    None,
                    lpsoft,
                    params
                )
        else:
            params['RandomHorizontallyFlip'] = False
        return img, mask, mask1, lpsoft, params


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                mask.transpose(Image.FLIP_TOP_BOTTOM),
            )
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            img.resize(self.size, Image.BILINEAR),
            mask.resize(self.size, Image.NEAREST),
        )


class RandomTranslate(object):
    def __init__(self, offset):
        self.offset = offset # tuple (delta_x, delta_y)

    def __call__(self, img, mask):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])
        
        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0
        
        cropped_img = tf.crop(img, 
                              y_crop_offset, 
                              x_crop_offset, 
                              img.size[1]-abs(y_offset), 
                              img.size[0]-abs(x_offset))

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)
        
        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)
        
        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)
        
        return (
              tf.pad(cropped_img, 
                     padding_tuple, 
                     padding_mode='reflect'),
              tf.affine(mask,
                        translate=(-x_offset, -y_offset),
                        scale=1.0,
                        angle=0.0,
                        shear=0.0,
                        fillcolor=250))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(img, 
                      translate=(0, 0),
                      scale=1.0, 
                      angle=rotate_degree, 
                      resample=Image.BILINEAR,
                      fillcolor=(0, 0, 0),
                      shear=0.0),
            tf.affine(mask, 
                      translate=(0, 0), 
                      scale=1.0, 
                      angle=rotate_degree, 
                      resample=Image.NEAREST,
                      fillcolor=250,
                      shear=0.0))



class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )

def MyScale(img, lbl, size):
    """scale

    img, lbl, longer size
    """
    if isinstance(img, np.ndarray):
        _img = Image.fromarray(img)
        _lbl = Image.fromarray(lbl)
    else:
        _img = img
        _lbl = lbl
    assert _img.size == _lbl.size
    # prop = 1.0 * _img.size[0]/_img.size[1]
    w, h = size
    # h = int(size / prop)
    _img = _img.resize((w, h), Image.BILINEAR)
    _lbl = _lbl.resize((w, h), Image.NEAREST)
    return np.array(_img), np.array(_lbl)

def Flip(img, lbl, prop):
    """
    flip img and lbl with probablity prop
    """
    if isinstance(img, np.ndarray):
        _img = Image.fromarray(img)
        _lbl = Image.fromarray(lbl)
    else:
        _img = img
        _lbl = lbl
    if random.random() < prop:
        _img.transpose(Image.FLIP_LEFT_RIGHT),
        _lbl.transpose(Image.FLIP_LEFT_RIGHT),
    return np.array(_img), np.array(_lbl)

def MyRotate(img, lbl, degree):
    """
    img, lbl, degree
    randomly rotate clockwise or anti-clockwise
    """
    if isinstance(img, np.ndarray):
        _img = Image.fromarray(img)
        _lbl = Image.fromarray(lbl)
    else:
        _img = img
        _lbl = lbl
    _degree = random.random()*degree
    
    flags = -1
    if random.random() < 0.5:
        flags = 1
    _img = _img.rotate(_degree * flags)
    _lbl = _lbl.rotate(_degree * flags)
    return np.array(_img), np.array(_lbl)

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask, mask1=None, lpsoft=None, params=None):
        assert img.size == mask.size
        if mask1 is not None:
            assert (img.size == mask1.size)

        if params is None:
            params = {}
            
        # 调整为固定大小而不是随机大小，以确保所有图像具有相同尺寸
        h, w = self.size, self.size
        params['RandomSized'] = (h, w)

        img, mask = (
            img.resize((w, h), Image.BILINEAR),
            mask.resize((w, h), Image.NEAREST),
        )
        if mask1 is not None:
            mask1 = mask1.resize((w, h), Image.NEAREST)
        if lpsoft is not None:
            # 检查并处理lpsoft的维度
            if lpsoft.dim() == 1:
                lpsoft = lpsoft.view(1, 1, -1)
            elif lpsoft.dim() == 2:
                lpsoft = lpsoft.unsqueeze(0)
            lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[h, w], mode='bilinear', align_corners=True)[0]

        return img, mask, mask1, lpsoft, params

class RandomZoom(object):
    """随机放大图像中的目标区域，特别适合小目标增强"""
    def __init__(self, zoom_range=(1.0, 1.5), p=0.5):
        self.zoom_range = zoom_range
        self.p = p

    def __call__(self, img, mask, mask1=None, lpsoft=None, params={}):
        if random.random() < self.p:
            w, h = img.size
            zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
            
            # 计算新的大小
            new_w = int(w / zoom_factor)
            new_h = int(h / zoom_factor)
            
            # 计算裁剪区域
            x1 = random.randint(0, w - new_w) if w > new_w else 0
            y1 = random.randint(0, h - new_h) if h > new_h else 0
            
            # 裁剪并调整大小
            cropped_img = img.crop((x1, y1, x1 + new_w, y1 + new_h))
            cropped_mask = mask.crop((x1, y1, x1 + new_w, y1 + new_h))
            
            resized_img = cropped_img.resize((w, h), Image.BILINEAR)
            resized_mask = cropped_mask.resize((w, h), Image.NEAREST)
            
            # 确保params是字典类型
            if params is None:
                params = {}
            
            params['RandomZoom'] = (zoom_factor, x1, y1, new_w, new_h)
            
            if lpsoft is not None:
                # 处理lpsoft
                lpsoft = lpsoft[:, y1:y1 + new_h, x1:x1 + new_w]
                lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[h, w], mode='bilinear', align_corners=True)[0]
            
            if mask1 is not None:
                cropped_mask1 = mask1.crop((x1, y1, x1 + new_w, y1 + new_h))
                resized_mask1 = cropped_mask1.resize((w, h), Image.NEAREST)
                return resized_img, resized_mask, resized_mask1, lpsoft, params
            else:
                return resized_img, resized_mask, None, lpsoft, params
        
        return img, mask, mask1, lpsoft, params

class ComplexRotate(object):
    """复杂旋转，包含旋转、缩放和平移的组合，提高模型对小目标的鲁棒性"""
    def __init__(self, degree=10, scale_range=(0.9, 1.1), translate_range=(-10, 10), p=0.5):
        self.degree = degree
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.p = p

    def __call__(self, img, mask, mask1=None, lpsoft=None, params={}):
        if random.random() < self.p:
            # 随机旋转角度
            rotate_degree = random.uniform(-self.degree, self.degree)
            # 随机缩放因子
            scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
            # 随机平移距离
            translate_x = random.randint(self.translate_range[0], self.translate_range[1])
            translate_y = random.randint(self.translate_range[0], self.translate_range[1])
            
            # 确保params是字典类型
            if params is None:
                params = {}
                
            params['ComplexRotate'] = (rotate_degree, scale_factor, translate_x, translate_y)
            
            transformed_img = tf.affine(img, 
                          translate=(translate_x, translate_y),
                          scale=scale_factor, 
                          angle=rotate_degree, 
                          resample=Image.BILINEAR,
                          fillcolor=(0, 0, 0),
                          shear=0.0)
                          
            transformed_mask = tf.affine(mask, 
                          translate=(translate_x, translate_y), 
                          scale=scale_factor, 
                          angle=rotate_degree, 
                          resample=Image.NEAREST,
                          fillcolor=250,
                          shear=0.0)
            
            if lpsoft is not None:
                # 对于lpsoft，我们需要更复杂的处理
                # 这里简化处理，实际应用中可能需要更精确的变换
                h, w = lpsoft.shape[1:]
                grid = F.affine_grid(
                    torch.tensor([[
                        [scale_factor * math.cos(math.radians(rotate_degree)), 
                         scale_factor * math.sin(math.radians(rotate_degree)), 
                         translate_x / w * 2],
                        [-scale_factor * math.sin(math.radians(rotate_degree)), 
                         scale_factor * math.cos(math.radians(rotate_degree)), 
                         translate_y / h * 2]
                    ]]).float(), 
                    size=torch.Size((1, lpsoft.shape[0], h, w)),
                    align_corners=True
                )
                lpsoft = F.grid_sample(lpsoft.unsqueeze(0), grid, align_corners=True)[0]
            
            if mask1 is not None:
                transformed_mask1 = tf.affine(mask1, 
                              translate=(translate_x, translate_y), 
                              scale=scale_factor, 
                              angle=rotate_degree, 
                              resample=Image.NEAREST,
                              fillcolor=250,
                              shear=0.0)
                return transformed_img, transformed_mask, transformed_mask1, lpsoft, params
            else:
                return transformed_img, transformed_mask, None, lpsoft, params
        
        return img, mask, mask1, lpsoft, params

class MedicalImageAugmentation(object):
    """医学图像特定的数据增强
    包括:
    1. 对比度增强
    2. 亮度调整
    3. 饱和度调整
    4. 色调调整
    5. 锐化
    6. 高斯噪声
    7. 运动模糊
    """
    def __init__(self, contrast_range=(0.8, 1.2), brightness_range=(0.8, 1.2),
                 saturation_range=(0.8, 1.2), hue_range=(-0.1, 0.1),
                 sharpness_range=(0.8, 1.2), noise_std=0.01, blur_prob=0.3):
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.sharpness_range = sharpness_range
        self.noise_std = noise_std
        self.blur_prob = blur_prob

    def __call__(self, img, mask):
        # 对比度增强
        contrast_factor = random.uniform(*self.contrast_range)
        img = tf.adjust_contrast(img, contrast_factor)
        
        # 亮度调整
        brightness_factor = random.uniform(*self.brightness_range)
        img = tf.adjust_brightness(img, brightness_factor)
        
        # 饱和度调整
        saturation_factor = random.uniform(*self.saturation_range)
        img = tf.adjust_saturation(img, saturation_factor)
        
        # 色调调整
        hue_factor = random.uniform(*self.hue_range)
        img = tf.adjust_hue(img, hue_factor)
        
        # 锐化
        sharpness_factor = random.uniform(*self.sharpness_range)
        img = tf.adjust_sharpness(img, sharpness_factor)
        
        # 添加高斯噪声
        if random.random() < 0.5:
            img = np.array(img)
            noise = np.random.normal(0, self.noise_std, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
        
        # 运动模糊
        if random.random() < self.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        return img, mask

class PolypSpecificAugmentation(object):
    """息肉特定的数据增强
    包括:
    1. 局部对比度增强
    2. 局部亮度调整
    3. 局部锐化
    4. 局部噪声
    """
    def __init__(self, local_contrast_range=(0.8, 1.2), local_brightness_range=(0.8, 1.2),
                 local_sharpness_range=(0.8, 1.2), local_noise_std=0.01):
        self.local_contrast_range = local_contrast_range
        self.local_brightness_range = local_brightness_range
        self.local_sharpness_range = local_sharpness_range
        self.local_noise_std = local_noise_std

    def __call__(self, img, mask):
        img = np.array(img)
        mask = np.array(mask)
        
        # 找到息肉区域
        polyp_mask = (mask == 1)
        if np.any(polyp_mask):
            # 局部对比度增强
            contrast_factor = random.uniform(*self.local_contrast_range)
            img[polyp_mask] = np.clip(img[polyp_mask] * contrast_factor, 0, 255)
            
            # 局部亮度调整
            brightness_factor = random.uniform(*self.local_brightness_range)
            img[polyp_mask] = np.clip(img[polyp_mask] * brightness_factor, 0, 255)
            
            # 局部锐化
            if random.random() < 0.5:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                img[polyp_mask] = cv2.filter2D(img[polyp_mask], -1, kernel)
            
            # 局部噪声
            if random.random() < 0.5:
                noise = np.random.normal(0, self.local_noise_std, img[polyp_mask].shape)
                img[polyp_mask] = np.clip(img[polyp_mask] + noise, 0, 255)
        
        return Image.fromarray(img), Image.fromarray(mask)
