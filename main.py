import os
import time
import shutil
import sys
import argparse
from net.loss import *
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup

from net.model import make_model
from dataset.dataloader import DatasetEval, DatasetTrain
from dataset.transforms import get_transforms
from core.utils import setup_system, Logger
from core.trainer import train, predict, extract_features_with_attention
from core.evaluate import evaluate

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def visualize_attention_test(model, dataloader, device, save_dir, num_samples=5, target_size=(384, 384)):
    """
    在测试模式下可视化注意力图
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break

            # 根据数据加载器的结构获取图像
            if len(batch) == 3:
                imgs, ids, paths = batch
            else:
                # 适应不同的数据格式
                imgs = batch[0]
                paths = batch[2] if len(batch) > 2 else [f"sample_{sample_count}"] * imgs.size(0)

            imgs = imgs.to(device)

            try:
                # 提取特征和注意力图
                if hasattr(model, 'module'):
                    features, attention_maps = model.module.extract_features_with_attention(
                        imgs, target_size=target_size)
                else:
                    features, attention_maps = model.extract_features_with_attention(
                        imgs, target_size=target_size)

                # 可视化每个样本
                for i in range(min(imgs.size(0), num_samples - sample_count)):
                    img = imgs[i].cpu()
                    att_map = attention_maps[i] if attention_maps is not None else None

                    if att_map is not None:
                        # 反归一化图像用于可视化
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_denorm = img * std + mean
                        img_denorm = torch.clamp(img_denorm, 0, 1)

                        # 保存注意力可视化
                        save_path = os.path.join(save_dir, f'test_sample_{sample_count + i + 1}.png')
                        save_attention_visualization(att_map, img_denorm, save_path)

                sample_count += imgs.size(0)

            except Exception as e:
                print(f"Error visualizing attention in test mode: {e}")
                continue

    print(f"Visualized {sample_count} test attention maps in {save_dir}")


def save_attention_visualization(attention_map, original_image, save_path, alpha=0.6):
    """
    保存注意力可视化结果
    """
    try:
        # 处理注意力图
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.detach().cpu().squeeze().numpy()

        # 归一化到 [0, 1]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # 应用颜色映射 (jet: 蓝色-青色-黄色-红色)
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 处理原始图像
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.detach().cpu().numpy()
            if original_image.shape[0] == 3:  # [C, H, W] -> [H, W, C]
                original_image = original_image.transpose(1, 2, 0)
            original_image = (original_image * 255).astype(np.uint8)

        # 调整热力图尺寸匹配原图
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

        print(f"Attention visualization saved to: {save_path}")

    except Exception as e:
        print(f"Error saving attention visualization: {e}")


@dataclass
class Configuration:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train and Test')
        # Model Config
        parser.add_argument('--backbone_name', default="convnext_tiny", type=str,
                            help='"convnext_tiny" | "convnext_base" | "convnext_large" | "convnext_xlarge"')
        parser.add_argument('--img_size', default=384, type=int, help='input image size')
        parser.add_argument('--only_test', default=False, type=bool, help='use pretrained net to test')
        parser.add_argument('--ckpt_path',
                            default='checkpoints/University-1652/convnext_tiny/D2S/weights_e3_0.9420.pth', type=str,
                            help='path to pretrained checkpoint file')

        # Training Config
        parser.add_argument('--mixed_precision', default=True, type=bool)
        parser.add_argument('--custom_sampling', default=True, type=bool)
        parser.add_argument('--seed', default=1, type=int, help='random seed')
        parser.add_argument('--epochs', default=10, type=int)
        parser.add_argument('--batch_size', default=32, type=int, help='remember the bs is for 2 branches')
        parser.add_argument('--verbose', default=True, type=bool)
        parser.add_argument('--gpu_ids', default=(0), type=tuple)

        # Eval Config
        parser.add_argument('--batch_size_eval', default=32, type=int)
        parser.add_argument('--eval_every_n_epoch', default=1, type=int)
        parser.add_argument('--normalize_features', default=True, type=bool)
        parser.add_argument('--eval_gallery_n', default=-1, type=int)

        # Optimizer Config
        parser.add_argument('--clip_grad', default=100.0, type=float)
        parser.add_argument('--grad_checkpointing', default=False, type=bool)

        # Learning Rate Config
        parser.add_argument('--lr', default=0.0001, type=float)
        parser.add_argument('--scheduler', default="cosine", type=str,
                            help=r'"polynomial" | "cosine" | "constant" | None')
        parser.add_argument('--warmup_epochs', default=0.1, type=float)
        parser.add_argument('--lr_end', default=0.0001, type=float)

        # Dataset Config
        parser.add_argument('--dataset', default='D2S', type=str, help="'D2S' | 'S2D'")
        parser.add_argument('--altitude', default=300, type=int, help="150|200|250|300")
        parser.add_argument('--data_folder', default='F:/lilaoshi/Dataset/CVGL', type=str)
        parser.add_argument('--dataset_name', default='SUES-200', type=str, help="'SUES-200' | 'University-1652'")

        # Augment Images Config
        parser.add_argument('--prob_flip', default=0.5, type=float,
                            help='flipping the sat image and drone image simultaneously')

        # Eval before training Config
        parser.add_argument('--zero_shot', default=False, type=bool)

        # Checkpoint to start from Config
        parser.add_argument('--checkpoint_start', default=None)

        # Visualization Config
        parser.add_argument('--visualize_attention', default=False, type=bool,
                            help='Whether to visualize attention maps')
        parser.add_argument('--visualize_freq', default=5, type=int, help='Frequency of visualization (epochs)')
        parser.add_argument('--visualize_samples', default=5, type=int, help='Number of samples to visualize')

        # Set num_workers to 0 if on Windows Config
        parser.add_argument('--num_workers', default=0 if os.name == 'nt' else 4, type=int)

        # Train on GPU if available Config
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

        # For better performance Config
        parser.add_argument('--cudnn_benchmark', default=True, type=bool)

        # Make cudnn deterministic Config
        parser.add_argument('--cudnn_deterministic', default=False, type=bool)

        args = parser.parse_args(namespace=self)


# -----------------------------------------------------------------------------#
# Train Config                                                                 #
# -----------------------------------------------------------------------------#
config = Configuration()

if config.dataset_name == 'SUES-200':
    config.num_classes = 200
    if config.dataset == 'D2S':
        config.query_folder_train = f'{config.data_folder}/{config.dataset_name}/Training/{config.altitude}/satellite'
        config.gallery_folder_train = f'{config.data_folder}/{config.dataset_name}/Training/{config.altitude}/drone'
        config.query_folder_test = f'{config.data_folder}/{config.dataset_name}/Testing/{config.altitude}/query_drone'
        config.gallery_folder_test = f'{config.data_folder}/{config.dataset_name}/Testing/{config.altitude}/gallery_satellite'
    elif config.dataset == 'S2D':
        config.query_folder_train = f'{config.data_folder}/{config.dataset_name}/Training/{config.altitude}/satellite'
        config.gallery_folder_train = f'{config.data_folder}/{config.dataset_name}/Training/{config.altitude}/drone'
        config.query_folder_test = f'{config.data_folder}/{config.dataset_name}/Testing/{config.altitude}/query_satellite'
        config.gallery_folder_test = f'{config.data_folder}/{config.dataset_name}/Testing/{config.altitude}/gallery_drone'

elif config.dataset_name == 'University-1652':
    config.num_classes = 701
    if config.dataset == 'D2S':
        config.query_folder_train = f'{config.data_folder}/{config.dataset_name}/train/satellite'
        config.gallery_folder_train = f'{config.data_folder}/{config.dataset_name}/train/drone'
        config.query_folder_test = f'{config.data_folder}/{config.dataset_name}/test/query_drone'
        config.gallery_folder_test = f'{config.data_folder}/{config.dataset_name}/test/gallery_satellite'
    elif config.dataset == 'S2D':
        config.query_folder_train = f'{config.data_folder}/{config.dataset_name}/train/satellite'
        config.gallery_folder_train = f'{config.data_folder}/{config.dataset_name}/train/drone'
        config.query_folder_test = f'{config.data_folder}/{config.dataset_name}/test/query_satellite'
        config.gallery_folder_test = f'{config.data_folder}/{config.dataset_name}/test/gallery_drone'

else:
    raise ValueError('Please select a valid dataset')

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')

    model_path = "checkpoints/{}/{}/{}".format(config.dataset_name, config.backbone_name, time.strftime("%y%m%d%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    shutil.copyfile(os.path.basename(__file__), "{}/main.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                        #
    # -----------------------------------------------------------------------------#

    model = make_model(config.backbone_name, config.num_classes)
    # print(net)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_size = (config.img_size, config.img_size)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                   #
    # -----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    # Train
    train_dataset = DatasetTrain(query_folder=config.query_folder_train,
                                 gallery_folder=config.gallery_folder_train,
                                 transforms_query=train_sat_transforms,
                                 transforms_gallery=train_drone_transforms,
                                 prob_flip=config.prob_flip,
                                 shuffle_batch_size=config.batch_size,
                                 )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    # Reference Satellite Images
    query_dataset_test = DatasetEval(data_folder=config.query_folder_test,
                                     mode="query",
                                     transforms=val_transforms,
                                     )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    # Query Ground Images Test
    gallery_dataset_test = DatasetEval(data_folder=config.gallery_folder_test,
                                       mode="gallery",
                                       transforms=val_transforms,
                                       sample_ids=query_dataset_test.get_sample_ids(),
                                       gallery_n=config.eval_gallery_n,
                                       )

    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)

    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    # -----------------------------------------------------------------------------#
    # Test Only                                                                    #
    # -----------------------------------------------------------------------------#
    if config.only_test:
        print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))
        best_score = 0

        checkpoint = torch.load(config.ckpt_path)

        model.load_state_dict(checkpoint, strict=False)

        model = model.to(config.device)

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

        # 测试模式下的注意力可视化
        if config.visualize_attention:
            print("\n{}[Visualizing Test Attention Maps]{}".format(20 * "-", 20 * "-"))
            attention_dir = os.path.join(model_path, "attention_visualization_test")
            visualize_attention_test(
                model,
                query_dataloader_test,
                config.device,
                attention_dir,
                num_samples=config.visualize_samples,
                target_size=img_size
            )

        sys.exit()

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2. ** 10)
    else:
        scaler = None

    # -----------------------------------------------------------------------------#
    # optimizer                                                                    #
    # -----------------------------------------------------------------------------#

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # -----------------------------------------------------------------------------#
    # Scheduler                                                                    #
    # -----------------------------------------------------------------------------#

    train_steps_per = len(train_dataloader)
    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    # -----------------------------------------------------------------------------#
    # Zero Shot                                                                    #
    # -----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30 * "-", "Zero Shot", 30 * "-"))

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

    # -----------------------------------------------------------------------------#
    # Shuffle                                                                      #
    # -----------------------------------------------------------------------------#
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    # -----------------------------------------------------------------------------#
    # Train                                                                        #
    # -----------------------------------------------------------------------------#

    start_epoch = 0
    best_score = 0
    best_ap_score = 0

    for epoch in range(1, config.epochs + 1):
        # 设置当前epoch，用于trainer中的可视化
        config.current_epoch = epoch

        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        # 定期可视化注意力图
        if (config.visualize_attention and
                (epoch % config.visualize_freq == 0 or epoch == config.epochs)):
            print("\n{}[Visualizing Attention Maps - Epoch {}]{}".format(20 * "-", epoch, 20 * "-"))
            attention_dir = os.path.join(model_path, f"attention_epoch_{epoch}")
            visualize_attention_test(
                model,
                query_dataloader_test,
                config.device,
                attention_dir,
                num_samples=config.visualize_samples,
                target_size=img_size
            )

        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))

            r1_test, ap_test = evaluate(config=config,
                                        model=model,
                                        query_loader=query_dataloader_test,
                                        gallery_loader=gallery_dataloader_test,
                                        ranks=[1, 5, 10],
                                        step_size=1000,
                                        cleanup=True)

            if r1_test >= best_score:
                best_score = r1_test
                best_ap_score = ap_test
                print("*** Best R@1: {:.4f} - AP: {:.4f} !! ***".format(best_score * 100, best_ap_score))

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(),
                               '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))