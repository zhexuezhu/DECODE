import torch,os
import pandas as pd
import numpy as np
from PIL import Image
import logging
import open_clip
import pickle

import cv2
from PIL import Image
import random
import numpy as np
import torch
import logging
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from scipy.optimize import fsolve


class DirectT:
    def __init__(self):
        pass
    def __call__(self,x,U=None):
        return x
    
class UniformBlur:
    def __init__(self,blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        img_np = np.array(img)
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_blur = cv2.GaussianBlur(img_np, (self.blur_kernel_size, self.blur_kernel_size), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_blur)
    
class FoveaBlur:
    def __init__(self, h, w, blur_kernel_size, curve_type='exp', *args, **kwargs):
        self.blur_kernel_size = blur_kernel_size
        self.mask = np.zeros((h,w), np.float32)
        
        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1-c
        edge_resolution = 0

        initial_guess = [1.0, 1.0]
        def equations(vars):
            t, r = vars
            eq1 = r * (t - np.sin(t)) - 1  # x = 1
            eq2 = -r * (1 - np.cos(t)) + 1.0  # y = 0
            return [eq1, eq2]
        solution = fsolve(equations, initial_guess)
        t_max, r_solution = solution
        self.r = r_solution

        fun_degrade = getattr(self, curve_type, None)
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1,distance/max_distance)
                y0 = fun_degrade(x0,**kwargs)
                self.mask[i, j] = edge_resolution + (center_resolution - edge_resolution) * y0

    def alphaBlend(self, img1, img2, mask):
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
        return blended
    
    def __call__(self, img, blur_kernel_size=None): 
        if blur_kernel_size ==None:
            blur_kernel_size = self.blur_kernel_size
        img = np.array(img)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        blured = cv2.GaussianBlur(img, (blur_kernel_size,blur_kernel_size), 0)
        blended = self.alphaBlend(img, blured, 1- self.mask)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended)
    
    def linear(self,x,**kwargs):
        return 1-x
    
    def exp(self,x,**kwargs):
        system_g = kwargs.get('system_g', 4)
        return  np.exp(-system_g * x)
    
    def quadratic(self,x,**kwargs):
        return  1 - x**2
    
    def log(self,x,**kwargs):
        b = 1/(np.e-1)
        a = np.log(b) + 1
        return  a - np.log(x + b)
    
    def brachistochrone(self,x,**kwargs):
        
        def equation(t):
            return t - np.sin(t) - (x / self.r)

        t0 = fsolve(equation, [1.0, 1.0])[0]
        y0 = -self.r * (1 - np.cos(t0)) + 1.0
        return  y0
        
        
class PerfectCenterFoveaBlur:
    """
    实现中心区域绝对清晰，边缘平滑过渡到模糊的“平顶”中央凹模糊。
    完全兼容原有 Pipeline 和 Uncertainty Aware 动态核机制。
    """
    def __init__(self, h, w, blur_kernel_size=11, clear_radius=40, decay_rate=5.0, *args, **kwargs):
        """
        :param h: 图像高度
        :param w: 图像宽度
        :param blur_kernel_size: 基础高斯模糊核大小
        :param clear_radius: 绝对清晰区域的圆形半径
        :param decay_rate: 边缘衰减速率，值越大越快达到极限模糊
        """
        self.blur_kernel_size = blur_kernel_size
        self.mask = np.zeros((h, w), dtype=np.float32)
        
        center_x, center_y = w // 2, h // 2
        # 计算从中心到图像四个角的最大可能距离
        max_distance = np.sqrt(max(center_x, w - center_x)**2 + max(center_y, h - center_y)**2)
        
        # 预计算掩码 (Mask)
        for i in range(h):
            for j in range(w):
                # 计算当前像素到中心的欧氏距离
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                
                if dist <= clear_radius:
                    # 中心清晰区：Alpha=1.0，完全保留原图
                    self.mask[i, j] = 1.0
                else:
                    # 边缘衰减区：平滑过渡，Alpha 指数级下降
                    normalized_dist = (dist - clear_radius) / (max_distance - clear_radius)
                    self.mask[i, j] = np.exp(-decay_rate * normalized_dist)

    def alphaBlend(self, img_raw, img_blur):
        """利用计算好的 mask 进行图像混合"""
        # 将单通道的 mask 扩展为 3 通道以便与 RGB/BGR 图像广播相乘
        alpha = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        
        # Alpha blending: 掩码越大，原图占比越高；掩码越小，模糊图占比越高
        blended = img_raw * alpha + img_blur * (1.0 - alpha)
        
        return cv2.convertScaleAbs(blended)

    def __call__(self, img, blur_kernel_size=None):
        # 兼容不确定性机制 (Uncertainty Aware) 传入的动态 kernel size
        if blur_kernel_size is None:
            blur_kernel_size = self.blur_kernel_size
            
        # OpenCV 要求高斯核必须是奇数，增加安全校验防止崩溃
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1

        # 兼容 PIL Image, PyTorch Tensor 和 NumPy Array
        if isinstance(img, torch.Tensor):
            from torchvision.transforms import functional as F
            img = F.to_pil_image(img)
            
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img.copy()
            
        # 确保输入转为 OpenCV 标准的 BGR 格式
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np

        # 生成全局模糊图
        img_blur = cv2.GaussianBlur(img_bgr, (blur_kernel_size, blur_kernel_size), 0)
        
        # 按照空间掩码进行融合
        blended = self.alphaBlend(img_bgr, img_blur)
        
        # 转回 RGB 供后续的 CLIP 图像编码器 (Vision Encoder) 使用
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended_rgb)