import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import torch
import math


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)
    
class EEGProjectLayer(nn.Module):
    def __init__(self,  z_dim,c_num, timesteps, drop_proj=0.6):
        super(EEGProjectLayer, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1]-self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x
    
class BaseModel(nn.Module):
    def __init__(self,  z_dim, c_num, timesteps, embedding_dim = 1440):
        super(BaseModel, self).__init__()

        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(z_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x

class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.Dropout(0.5),
            )
    
class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps,embedding_dim = 1400)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 25, (1, 10), (1, 1)),
                nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(25, 50, (1, 10), (1, 1)),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(50, 100, (1, 10), (1, 1)),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(100, 200, (1, 10), (1, 1)),
                nn.BatchNorm2d(200),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
            )
        
class EEGnet(BaseModel):
    def __init__(self,  z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim = 1248)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                # nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout2d(0.5)
            )
        
class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )

"""class Proj_text(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
"""
class PatchEmbedding(nn.Module):
    """Patch嵌入层，将EEG信号转换为嵌入向量"""
    def __init__(self, emb_size=40, c_num=17):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 1, c_num, time_length]
        x = self.tsconv(x)
        x = self.projection(x)
        return x

class PositionalEncoding(nn.Module):
    """位置编码层，为时间序列添加位置信息"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 为偶数和奇数d_model分别处理
        sin_indices = torch.arange(0, d_model, 2)
        cos_indices = torch.arange(1, d_model, 2)
        
        pe[:, sin_indices] = torch.sin(position * div_term[:len(sin_indices)])
        pe[:, cos_indices] = torch.cos(position * div_term[:len(cos_indices)])
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 形状: (time_length, batch_size, channels)
        T, B, C = x.shape
        
        # 生成位置编码，确保形状匹配
        pe = self.pe[:T, :C]  # (T, C) - 只使用前C个通道的位置编码
        pe = pe.unsqueeze(1).repeat(1, B, 1)  # (T, B, C)
        
        x = x + pe
        return x

class EEGAttention(nn.Module):
    """EEG注意力机制，使用Transformer编码器处理时间序列"""
    def __init__(self, channel, nhead):
        super().__init__()
        self.pos_encoder = PositionalEncoding(channel)  # 使用channel作为d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=channel, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = channel  # 确保d_model等于channel

    def forward(self, src):
        src = src.permute(2, 0, 1)  # [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # [batch_size, channel, time_length]

class Enc_eeg(nn.Sequential):
    """EEG编码器"""
    def __init__(self, emb_size=40, c_num=17):
        super().__init__(
            PatchEmbedding(emb_size, c_num),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    """EEG特征投影层，用于与其他模态对齐"""
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class Cogcap(nn.Module):
    """
    主EEG特征提取模型
    """
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__()
        self.attention_model = EEGAttention(c_num, nhead=1)
        # 移除未使用的subject_wise_linear层
        self.enc_eeg = Enc_eeg(c_num=c_num)
        self.proj_eeg = Proj_eeg(proj_dim=z_dim)
        self.timesteps = timesteps
        self.c_num = c_num
        # 添加必要的属性，确保与其他模型兼容
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        # 1. 处理不同维度的输入
        if x.dim() == 4:
            # (batch, trials, channels, timesteps) -> (batch, channels, timesteps)
            x = x.mean(dim=1)
        elif x.dim() == 2:
            # (batch, channels*timesteps) -> (batch, channels, timesteps)
            x = x.view(x.size(0), self.c_num, -1)
        
        # 2. 确保通道维度在前
        if x.dim() == 3 and x.size(1) > x.size(2):
            # (batch, timesteps, channels) -> (batch, channels, timesteps)
            x = x.transpose(1, 2)
        
        # 3. 选择时间窗口
        time_length = self.timesteps[1] - self.timesteps[0]
        x = x[:, :, self.timesteps[0]:self.timesteps[1]]  # (batch, c_num, time_length)
        
        # 4. 执行注意力变换
        x = self.attention_model(x)
        # 移除未使用的subject_wise_linear层
        
        # 5. 编码和投影
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        return out
        
class EEGTransNet(nn.Module):
    """
    完整的EEGTransNet网络
    输入: [batch, channels, timesteps]
    输出: [batch, z_dim]
    """
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__()
        self.c_num = c_num
        self.timesteps = timesteps[1] - timesteps[0]  # 实际使用的时间步数
        
        # Stage 1: 多尺度时间特征提取
        self.temporal_feature_extractor = nn.Sequential(
            MultiScaleTemporalConv(in_channels=c_num, out_channels=128),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            MultiScaleTemporalConv(in_channels=128, out_channels=256),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            MultiScaleTemporalConv(in_channels=256, out_channels=512),
            nn.AdaptiveAvgPool1d(32)  # 固定输出时间长度为32
        )
        
        # Stage 2: 时序Transformer编码
        self.temporal_encoder = TemporalTransformer(
            d_model=512,
            nhead=8,
            num_layers=3
        )
        
        # Stage 3: 残差投影头
        self.projection_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            
            ResidualProjectionBlock(1024, 1024),
            ResidualProjectionBlock(1024, 1024),
            
            nn.Linear(1024, z_dim),
            nn.LayerNorm(z_dim)
        )
        
        # 对比学习参数
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        """
        前向传播
        x: EEG信号 [batch, channels, timesteps]
        """
        # Stage 1: 多尺度时间特征提取
        temporal_features = self.temporal_feature_extractor(x)
        # temporal_features: [batch, 512, 32]
        
        # Stage 2: Transformer时序编码
        temporal_features = temporal_features.transpose(1, 2)  # [batch, 32, 512]
        encoded_features = self.temporal_encoder(temporal_features)
        # encoded_features: [batch, 512]
        
        # Stage 3: 投影头
        projected_features = self.projection_head(encoded_features)
        # projected_features: [batch, z_dim]
        
        return projected_features

# 在原有文件中添加以下辅助类
class MultiScaleTemporalConv(nn.Module):
    """多尺度时间卷积块"""
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        
        # 三个不同尺度的时间卷积
        self.conv_large = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, 31, padding=15),
            nn.BatchNorm1d(out_channels//4),
            nn.ELU(),
            nn.Dropout(0.2)
        )
        
        self.conv_medium = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, 15, padding=7),
            nn.BatchNorm1d(out_channels//4),
            nn.ELU(),
            nn.Dropout(0.2)
        )
        
        self.conv_small = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, 7, padding=3),
            nn.BatchNorm1d(out_channels//4),
            nn.ELU(),
            nn.Dropout(0.2)
        )
        
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels//4 * 3, out_channels//8, 1),
            nn.ELU(),
            nn.Conv1d(out_channels//8, out_channels//4 * 3, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(out_channels//4 * 3, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ELU()
        )
    
    def forward(self, x):
        # x: [batch, channels, timesteps]
        
        # 多尺度特征提取
        large_feat = self.conv_large(x)
        medium_feat = self.conv_medium(x)
        small_feat = self.conv_small(x)
        
        # 拼接特征
        concat_feat = torch.cat([large_feat, medium_feat, small_feat], dim=1)
        
        # 通道注意力
        weights = self.channel_attention(concat_feat)
        weighted_feat = concat_feat * weights
        
        # 特征融合
        output = self.fusion_conv(weighted_feat)
        
        return output

class PositionalEncoding1D(nn.Module):
    """1D位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class TemporalTransformer(nn.Module):
    """时序Transformer"""
    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding1D(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size = x.size(0)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 添加[CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer处理
        x = self.transformer(x)
        
        # 返回[CLS] token作为序列表示
        return x[:, 0, :]

class ResidualProjectionBlock(nn.Module):
    """残差投影块"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim, out_dim)
        )
        
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        out = self.layer_norm(out + residual)
        return out


class MixedEncoder(nn.Module):
    """
    单一共享 backbone + 双投影头，同时适配图像和文本监督。
    训练时联合优化两个对比损失。
    """
    def __init__(self, z_dim, c_num=63, timesteps=[0, 250], drop_proj=0.5):
        super().__init__()
        # 共享的 EEG 特征提取器（使用 Cogcap）
        self.backbone = Cogcap(z_dim=z_dim, c_num=c_num, timesteps=timesteps)

        # 图像模态投影头
        self.proj_img = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(z_dim),
        )

        # 文本模态投影头
        self.proj_text = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(z_dim),
        )

        # 两个独立的温度系数（可学习）
        self.logit_scale_img = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: [batch, channels, time]
        shared = self.backbone(x)                  # [batch, z_dim]
        z_img = F.normalize(self.proj_img(shared), dim=-1)
        z_text = F.normalize(self.proj_text(shared), dim=-1)
        return z_img, z_text
        