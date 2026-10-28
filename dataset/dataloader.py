import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random


def get_data(path):
    """获取数据目录结构"""
    data = {}
    print(f"Scanning directory: {path}")

    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"ERROR: Path does not exist: {path}")
        return data

    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            data[name] = {"path": dir_path}

            # 获取该目录下的所有图像文件
            image_files = []
            for file in os.listdir(dir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(file)

            data[name]["files"] = image_files
            print(f"Found {len(image_files)} images in {name}")

    return data


class DatasetTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        print(f"Initializing DatasetTrain with:")
        print(f"  Query folder: {query_folder}")
        print(f"  Gallery folder: {gallery_folder}")

        # 获取数据
        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)

        print(f"Found {len(self.query_dict)} query identities")
        print(f"Found {len(self.gallery_dict)} gallery identities")

        # 使用同时存在于gallery和query中的文件夹
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))

        if len(self.ids) == 0:
            print("WARNING: No common identities found between query and gallery!")
            # 检查各自有哪些ID
            print(f"Query IDs: {list(self.query_dict.keys())}")
            print(f"Gallery IDs: {list(self.gallery_dict.keys())}")
        else:
            print(f"Found {len(self.ids)} common identities")

        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:
            # 检查query目录是否有文件
            if len(self.query_dict[idx]["files"]) == 0:
                print(f"WARNING: No images found for query ID {idx}")
                continue

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]

            if len(gallery_imgs) == 0:
                print(f"WARNING: No images found for gallery ID {idx}")
                continue

            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                self.pairs.append((idx, label, query_img, "{}/{}".format(gallery_path, g)))

        print(f"Created {len(self.pairs)} training pairs")

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, gallery_img_path = self.samples[index]

        # 读取query图像
        if not os.path.exists(query_img_path):
            print(f"ERROR: Query image not found: {query_img_path}")
            # 返回黑色图像作为占位符
            query_img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            query_img = cv2.imread(query_img_path)
            if query_img is None:
                print(f"ERROR: Failed to load query image: {query_img_path}")
                query_img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # 读取gallery图像
        if not os.path.exists(gallery_img_path):
            print(f"ERROR: Gallery image not found: {gallery_img_path}")
            gallery_img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            gallery_img = cv2.imread(gallery_img_path)
            if gallery_img is None:
                print(f"ERROR: Failed to load gallery image: {gallery_img_path}")
                gallery_img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        # 随机水平翻转
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # 图像变换
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        """自定义shuffle函数，用于在batch中采样唯一的class_id"""

        print("\nShuffle Dataset:")

        # 检查是否有样本可以shuffle
        if len(self.pairs) == 0:
            print("WARNING: No pairs available for shuffling!")
            self.samples = []
            return

        pair_pool = copy.deepcopy(self.pairs)

        # 打乱pair顺序
        random.shuffle(pair_pool)

        # 记录在epoch中已经使用过的pair
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm(total=len(self.pairs))

        while True:
            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                idx, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)
                    break_counter = 0
                else:
                    # 如果pair不适合当前batch且在当前epoch中未使用过，则放回pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1

                if break_counter >= 512:
                    break
            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # 将当前batch添加到batches中
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # 等待一段时间再关闭进度条
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))

        # 添加安全检查，避免空列表
        if len(self.samples) > 0:
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))
        else:
            print("WARNING: No samples after shuffling!")


class DatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        print(f"Initializing DatasetEval with:")
        print(f"  Data folder: {data_folder}")
        print(f"  Mode: {mode}")

        self.data_dict = get_data(data_folder)

        # 使用所有找到的文件夹
        self.ids = list(self.data_dict.keys())

        if len(self.ids) == 0:
            print(f"WARNING: No identities found in {data_folder}")

        self.transforms = transforms
        self.given_sample_ids = sample_ids
        self.images = []
        self.sample_ids = []
        self.mode = mode
        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):
            # 检查是否有图像文件
            if len(self.data_dict[sample_id]["files"]) == 0:
                print(f"WARNING: No images found for ID {sample_id}")
                continue

            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                # 如果设置了gallery_n，限制每个identity的图像数量
                if self.gallery_n > 0 and j >= self.gallery_n:
                    break

                img_path = "{}/{}".format(self.data_dict[sample_id]["path"], file)
                self.images.append(img_path)
                self.sample_ids.append(sample_id)

        print(f"Loaded {len(self.images)} images for evaluation")

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        # 读取图像
        if not os.path.exists(img_path):
            print(f"ERROR: Image not found: {img_path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.imread(img_path)
            if img is None:
                print(f"ERROR: Failed to load image: {img_path}")
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 图像变换
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label, img_path

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)