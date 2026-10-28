import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models import create_model

from .DCA import DynamicConvolutionalAggregator
from .SFG import SpatialFocusGenerator


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate,
                 relu=False, bnorm=True, num_bottleneck=512, linear=True):
        super().__init__()

        layers = []
        if linear:
            layers.append(nn.Linear(input_dim, num_bottleneck))
        else:
            num_bottleneck = input_dim
        if bnorm:
            layers.append(nn.BatchNorm1d(num_bottleneck))
        if relu:
            layers.append(nn.LeakyReLU(0.1))
        if droprate > 0:
            layers.append(nn.Dropout(p=droprate))

        self.add_block = nn.Sequential(*layers)
        self.add_block.apply(weights_init)

        self.classifier = nn.Sequential(nn.Linear(num_bottleneck, class_num))
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class FineGrained(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dca = DynamicConvolutionalAggregator(
            k=3, inplanes=in_channels, planes=out_channels // 4
        )
        self.sfg = SpatialFocusGenerator(in_channels)

    def forward(self, x, return_attention=False, target_size=None):
        patch_feat = x.flatten(2)  # [B, C, H*W]

        if return_attention:
            # 返回注意力图
            att_feat, attention_map = self.sfg(x, return_attention=True, target_size=target_size)
        else:
            att_feat = self.sfg(x)  # [B, C, H, W]
            attention_map = None

        aligned_feat = self.dca(att_feat, patch_feat)  # [B, C, L]

        if return_attention:
            return aligned_feat, att_feat, attention_map
        else:
            return aligned_feat, att_feat


class CoarseGrained(nn.Module):

    def __init__(self, gem_p=3.0, dropout_p=0.1, feature_dim=768):
        super().__init__()
        self.gem_p = gem_p
        self.bn = nn.BatchNorm1d(feature_dim)
        self.dropout = nn.Dropout(dropout_p)

    @staticmethod
    def gem_pool(x, p=3, eps=1e-6):
        """Generalized Mean Pooling (GeM)"""
        return F.avg_pool2d(
            x.clamp(min=eps).pow(p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / p)

    def forward(self, x):
        f_vec = self.gem_pool(x, self.gem_p).flatten(1)  # [B, C]
        f_vec = self.bn(f_vec)
        f_vec = self.dropout(f_vec)
        f_vec = F.normalize(f_vec, p=2, dim=1)
        return f_vec


class C2FNet(nn.Module):
    """C2FNet: Backbone + Coarse/Fine Feature Branches"""

    def __init__(self, backbone_name="convnext_tiny", num_classes=701,
                 gem_p=3.0, dropout_p=0.1):
        super().__init__()

        # Backbone
        self.backbone = create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool=""
        )
        if 'convnext_tiny' in backbone_name:
            self.feature_dim = 768
        elif 'convnext_base' in backbone_name:
            self.feature_dim = 1024
        elif 'convnext_large' in backbone_name:
            self.feature_dim = 1536
        elif 'convnext_xlarge' in backbone_name:
            self.feature_dim = 2048
        else:
            raise ValueError(f"{backbone_name} is not supported")

        # Modules
        self.coarse_grained = CoarseGrained(gem_p, dropout_p, self.feature_dim)
        self.coarse_grained_aux = CoarseGrained(gem_p, dropout_p, self.feature_dim)
        self.fine_grained = FineGrained(self.feature_dim, self.feature_dim)

        # Classifiers
        self.classifier = ClassBlock(self.feature_dim, num_classes, droprate=0.5)
        self.classifier_aux = ClassBlock(self.feature_dim, num_classes, droprate=0.5)

        # Learnable logit scale
        self.logit_scale_blocks = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, return_attention=False, target_size=None):
        """
        前向传播

        Args:
            x: 输入图像
            return_attention: 是否返回注意力图
            target_size: 注意力图上采样的目标尺寸
        """
        features = self.backbone.forward_features(x)

        f_c = self.coarse_grained(features)

        if return_attention:
            f_f, f_a, attention_map = self.fine_grained(features, return_attention=True, target_size=target_size)
        else:
            f_f, f_a = self.fine_grained(features)
            attention_map = None

        f_a = self.coarse_grained_aux(f_a)

        cls = self.classifier(f_c)
        cls_aux = self.classifier_aux(f_a)

        if return_attention:
            return f_c, f_f, cls, cls_aux, attention_map
        else:
            return f_c, f_f, cls, cls_aux

    def forward_with_attention(self, x, target_size=None):
        """
        专门用于获取注意力的前向传播
        """
        return self.forward(x, return_attention=True, target_size=target_size)

    def extract_features_with_attention(self, x, target_size=None):
        """
        提取特征和注意力图
        """
        features = self.backbone.forward_features(x)
        f_c = self.coarse_grained(features)
        f_f, f_a, attention_map = self.fine_grained(features, return_attention=True, target_size=target_size)

        return f_c, attention_map

    def extract_satellite_features(self, x):
        """提取卫星图像特征（用于trainer）"""
        return self.forward(x)

    def extract_drone_features(self, x):
        """提取无人机图像特征（用于trainer）"""
        return self.forward(x)


def make_model(backbone_name, num_classes):
    return C2FNet(backbone_name, num_classes=num_classes, gem_p=3.0, dropout_p=0.1)