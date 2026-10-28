import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
from net.loss import *
import torch.nn as nn
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def visualize_attention_during_training(model, batch_data, epoch, batch_idx, save_dir, num_samples=3):
    """
    在训练过程中可视化注意力图
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        query, reference, ids, labels = batch_data

        # 只取前几个样本进行可视化
        num_viz = min(num_samples, query.size(0))

        for i in range(num_viz):
            # 获取单个样本
            query_img = query[i:i + 1].to(next(model.parameters()).device)

            try:
                # 尝试使用支持注意力返回的前向传播
                if hasattr(model, 'module'):
                    # DataParallel包装的模型
                    if hasattr(model.module, 'forward_with_attention'):
                        outputs, attention_maps = model.module.forward_with_attention(query_img)
                    else:
                        # 如果模型没有专门的方法，使用普通forward
                        outputs = model.module(query_img)
                        attention_maps = None
                else:
                    if hasattr(model, 'forward_with_attention'):
                        outputs, attention_maps = model.forward_with_attention(query_img)
                    else:
                        outputs = model(query_img)
                        attention_maps = None

                # 如果有注意力图，进行可视化
                if attention_maps is not None:
                    # 反归一化图像用于可视化
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(query_img.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(query_img.device)
                    img_denorm = query_img * std + mean
                    img_denorm = torch.clamp(img_denorm, 0, 1)

                    # 假设attention_maps是最后一个SFG层的输出
                    if isinstance(attention_maps, list) and len(attention_maps) > 0:
                        att_map = attention_maps[-1]  # 取最后一个注意力图
                    else:
                        att_map = attention_maps

                    # 保存注意力可视化
                    save_path = os.path.join(save_dir, f'epoch_{epoch}_batch_{batch_idx}_sample_{i}.png')
                    save_attention_visualization_simple(att_map, img_denorm[0], save_path)

            except Exception as e:
                print(f"Error in attention visualization: {e}")
                continue

    model.train()


def save_attention_visualization_simple(attention_map, original_image, save_path, alpha=0.6):
    """
    简化的注意力可视化保存函数
    """
    try:
        # 处理注意力图
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.detach().cpu().squeeze().numpy()

        # 归一化到 [0, 1]
        if attention_map.max() > attention_map.min():
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # 处理原始图像
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.detach().cpu().numpy()
            if original_image.shape[0] == 3:  # [C, H, W] -> [H, W, C]
                original_image = original_image.transpose(1, 2, 0)
            original_image = (original_image * 255).astype(np.uint8)

        # 调整热力图尺寸匹配原图
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if heatmap.shape[:2] != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

        # 叠加热力图和原图
        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 热力图
        axes[1].imshow(heatmap)
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')

        # 叠加图
        axes[2].imshow(overlayed)
        axes[2].set_title('Overlayed Result')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error saving attention visualization: {e}")


def train(train_config, model, dataloader, optimizer, scheduler=None, scaler=None):
    # set net train mode
    model.train()
    losses = AverageMeter()

    loss_info = DSA_loss(True)

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    # 创建注意力可视化目录
    attention_dir = None
    if hasattr(train_config, 'visualize_attention') and train_config.visualize_attention:
        model_path = "checkpoints/{}/{}/{}".format(
            getattr(train_config, 'dataset_name', 'default'),
            getattr(train_config, 'backbone_name', 'default'),
            time.strftime("%y%m%d%H%M%S")
        )
        attention_dir = os.path.join(model_path, "attention_visualization_train")
        os.makedirs(attention_dir, exist_ok=True)

    # for loop over one epoch
    for batch_idx, (query, reference, ids, labels) in enumerate(bar):

        # 在特定批次可视化注意力
        if (attention_dir is not None and
                hasattr(train_config, 'visualize_attention') and
                train_config.visualize_attention and
                batch_idx % 50 == 0):  # 每50个batch可视化一次
            visualize_attention_during_training(
                model,
                (query, reference, ids, labels),
                getattr(train_config, 'current_epoch', 1),
                batch_idx,
                attention_dir,
                num_samples=min(3, query.size(0))
            )

        if scaler:
            with autocast():  # 使用混合精度
                # data (batches) to device
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                labels = labels.to(train_config.device)

                # Forward pass - 训练时不返回注意力
                out_11, out_12, cls_11, cls_12 = model(query)
                out_21, out_22, cls_21, cls_22 = model(reference)

                loss = loss_con(out_11, out_21) + 0.1 * loss_info(out_12, out_22, model.logit_scale_blocks.exp())

                if labels is not None:
                    loss_cls = loss_ce(cls_11, labels) + loss_ce(cls_12, labels) + loss_ce(cls_21, labels) + loss_ce(
                        cls_22, labels)
                    loss = loss + 0.1 * loss_cls
                losses.update(loss.item())

            scaler.scale(loss).backward()  # 混合精度反向传播

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

            # Update net parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:
            # data (batches) to device
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)
            labels = labels.to(train_config.device)

            # Forward pass - 训练时不返回注意力
            out_11, out_12, cls_11, cls_12 = model(query)
            out_21, out_22, cls_21, cls_22 = model(reference)

            loss = loss_con(out_11, out_21) + 0.1 * loss_info(out_12, out_22, model.logit_scale_blocks.exp())

            if labels is not None:
                loss_cls = loss_ce(cls_11, labels) + loss_ce(cls_12, labels) + loss_ce(cls_21, labels) + loss_ce(cls_22,
                                                                                                                 labels)
                loss = loss + 0.1 * loss_cls
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

            # Update net parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:
            monitor = {
                "loss": "{:.4f}".format(loss.item()),
                "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader, return_attention=False):
    """
    修改predict函数以支持返回注意力图
    """
    model.eval()

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []
    path_list = []
    ids_list = []
    attention_maps_list = []  # 新增：存储注意力图

    with torch.no_grad():
        for img, ids, img_path in bar:
            ids_list.append(ids)
            path_list.extend(img_path)

            with autocast():
                img = img.to(train_config.device)

                # 根据是否返回注意力图调整前向传播
                if return_attention:
                    try:
                        if hasattr(model, 'module'):
                            if hasattr(model.module, 'forward_with_attention'):
                                img_feature, _, _, _, attention_maps = model.module.forward_with_attention(img)
                            else:
                                img_feature, _, _, _ = model.module(img)
                                attention_maps = None
                        else:
                            if hasattr(model, 'forward_with_attention'):
                                img_feature, _, _, _, attention_maps = model.forward_with_attention(img)
                            else:
                                img_feature, _, _, _ = model(img)
                                attention_maps = None
                    except:
                        # 如果前向传播失败，回退到普通模式
                        img_feature, _, _, _ = model(img)
                        attention_maps = None
                else:
                    img_feature, _, _, _ = model(img)
                    attention_maps = None

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

            # 保存注意力图
            if return_attention and attention_maps is not None:
                attention_maps_list.append(attention_maps.cpu())

        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)

        # 合并注意力图
        if return_attention and attention_maps_list:
            attention_maps = torch.cat(attention_maps_list, dim=0)
        else:
            attention_maps = None

    if train_config.verbose:
        bar.close()

    if return_attention:
        return img_features, ids_list, path_list, attention_maps
    else:
        return img_features, ids_list, path_list


def extract_features_with_attention(model, dataloader, device, normalize=True):
    """
    专门用于提取特征和注意力图的函数
    """
    model.eval()

    features_list = []
    attention_maps_list = []
    paths_list = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                imgs, ids, paths = batch
            else:
                imgs, paths = batch[0], batch[2]  # 适应不同的数据格式

            imgs = imgs.to(device)

            try:
                # 尝试使用支持注意力的前向传播
                if hasattr(model, 'module'):
                    if hasattr(model.module, 'extract_features_with_attention'):
                        features, attention_maps = model.module.extract_features_with_attention(imgs)
                    else:
                        features, _, _, _ = model.module(imgs)
                        attention_maps = None
                else:
                    if hasattr(model, 'extract_features_with_attention'):
                        features, attention_maps = model.extract_features_with_attention(imgs)
                    else:
                        features, _, _, _ = model(imgs)
                        attention_maps = None

                if normalize:
                    features = F.normalize(features, dim=-1)

                features_list.append(features.cpu())
                if attention_maps is not None:
                    attention_maps_list.append(attention_maps.cpu())
                paths_list.extend(paths)

            except Exception as e:
                print(f"Error extracting features with attention: {e}")
                continue

    features = torch.cat(features_list, dim=0) if features_list else None
    attention_maps = torch.cat(attention_maps_list, dim=0) if attention_maps_list else None

    return features, attention_maps, paths_list