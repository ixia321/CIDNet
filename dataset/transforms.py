import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from albumentations.core.transforms_interface import ImageOnlyTransform
import imgaug.augmenters as iaa

# ********************************************** apply multi-weather setting **********************************************************

class ImgAugTransform(ImageOnlyTransform):
    def __init__(self, aug, always_apply=False, p=1.0):
        super(ImgAugTransform, self).__init__(always_apply, p)
        self.aug = aug

    def apply(self, img, **params):
        return self.aug(image=img)


# 自定义云层变换
class CustomCloudLayer(ImgAugTransform):
    def __init__(self, intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2,
                 alpha_min=1.0, alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2,
                 sparsity=0.9, density_multiplier=0.5, seed=None, always_apply=False, p=1.0):
        aug = iaa.CloudLayer(
            intensity_mean=intensity_mean,
            intensity_freq_exponent=intensity_freq_exponent,
            intensity_coarse_scale=intensity_coarse_scale,
            alpha_min=alpha_min,
            alpha_multiplier=alpha_multiplier,
            alpha_size_px_max=alpha_size_px_max,
            alpha_freq_exponent=alpha_freq_exponent,
            sparsity=sparsity,
            density_multiplier=density_multiplier,
            seed=seed
        )
        super(CustomCloudLayer, self).__init__(aug, always_apply, p)


# 自定义雨变换
class CustomRain(ImgAugTransform):
    def __init__(self, drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=None, always_apply=False, p=1.0):
        aug = iaa.Rain(
            drop_size=drop_size,
            speed=speed,
            seed=seed
        )
        super(CustomRain, self).__init__(aug, always_apply, p)


# 自定义雪花变换
class CustomSnowflakes(ImgAugTransform):
    def __init__(self, flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=None, always_apply=False, p=1.0):
        aug = iaa.Snowflakes(
            flake_size=flake_size,
            speed=speed,
            seed=seed
        )
        super(CustomSnowflakes, self).__init__(aug, always_apply, p)


iaa_weather_list = [

    # 0. Normal
    A.NoOp(),
    # 1. Fog
    A.Compose([
        CustomCloudLayer()
    ]),
    # 2. Rain
    A.Compose([
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
    ]),
    # 3. Snow
    A.Compose([
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
    ]),
    # 4. Fog+Rain
    A.Compose([
        CustomCloudLayer(),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)
    ]),
    # 5. Fog+Snow
    A.Compose([
        CustomCloudLayer(),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)
    ]),
    # 6. Rain+Snow
    A.Compose([
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
    ]),
    # 7. Dark
    A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(9, 11), p=0.5),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.5),
        ]),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.15), contrast_limit=(0.2, 0.2), p=1)
    ]),
    # 8. Over-exposure
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0, 0.3), contrast_limit=(1.3, 1.6), p=1)
    ]),
    # 9. Wind
    A.Compose([
        A.MotionBlur(blur_limit=15, p=1)
    ])
]


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   weather_id=0):
    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                # Multi-weather settings.
                                # iaa_weather_list[weather_id],

                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                      # Multi-weather settings.
                                      # A.OneOf(iaa_weather_list, p=1.0),

                                      A.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.3,
                                                    always_apply=False, p=0.5),
                                      A.OneOf([
                                          A.AdvancedBlur(p=1.0),
                                          A.Sharpen(p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                          A.GridDropout(ratio=0.4, p=1.0),
                                          A.CoarseDropout(max_holes=25,
                                                          max_height=int(0.2 * img_size[0]),
                                                          max_width=int(0.2 * img_size[0]),
                                                          min_holes=10,
                                                          min_height=int(0.1 * img_size[0]),
                                                          min_width=int(0.1 * img_size[0]),
                                                          p=1.0),
                                      ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])

    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                        # Multi-weather U1652 settings.
                                        # A.OneOf(iaa_weather_list, p=1.0),

                                        A.ColorJitter(brightness=0.15, contrast=0.7, saturation=0.3, hue=0.3,
                                                      always_apply=False, p=0.5),
                                        A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                        ], p=0.3),
                                        A.OneOf([
                                            A.GridDropout(ratio=0.4, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2 * img_size[0]),
                                                            max_width=int(0.2 * img_size[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1 * img_size[0]),
                                                            min_width=int(0.1 * img_size[0]),
                                                            p=1.0),
                                        ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])

    return val_transforms, train_sat_transforms, train_drone_transforms