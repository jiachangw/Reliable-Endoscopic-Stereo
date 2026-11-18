import os
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate
from stereo.utils.common_utils import get_pos_fullres
import cv2

# class RigidEndoDataset(DatasetTemplate):
#     def __init__(self, data_info, data_cfg, mode):
#         super().__init__(data_info, data_cfg, mode)
#         self.scared_root = self.data_info.get('EVAL_DATA_PATH', self.root)

#     def __getitem__(self, idx):
#         item = self.data_list[idx]
#         if self.mode == 'evaluating':
#             full_paths = [os.path.join(self.scared_root, x) for x in item[0:3]]
#             left_img_path, right_img_path, disp_img_path = full_paths
#             left_img = Image.open(left_img_path).convert('RGB')
#             left_img = np.array(left_img, dtype=np.float32)
#             right_img = Image.open(right_img_path).convert('RGB')
#             right_img = np.array(right_img, dtype=np.float32)
#             disp_img = cv2.imread(disp_img_path, cv2.IMREAD_UNCHANGED) / 128
#             disp_img = np.array(disp_img, dtype=np.float32)
#             assert not np.isnan(disp_img).any(), 'disp_img has nan'
#             sample = {
#                 'left': left_img,  # [H, W, 3]
#                 'right': right_img,  # [H, W, 3]
#                 'disp': disp_img,  # [H, W]
#             }
#         else:

#             full_paths = [os.path.join(self.root, x) for x in item[0:3]]
#             left_img_path, right_img_path, disp_img_path = full_paths
#             left_img = Image.open(left_img_path).convert('RGB')
#             left_img = np.array(left_img, dtype=np.float32)
#             right_img = Image.open(right_img_path).convert('RGB')
#             right_img = np.array(right_img, dtype=np.float32)
#             disp_img = rgb_to_float(cv2.imread(disp_img_path))
#             assert not np.isnan(disp_img).any(), 'disp_img has nan'
#             sample = {
#                 'left': left_img,  # [H, W, 3]
#                 'right': right_img,  # [H, W, 3]
#                 'disp': disp_img,  # [H, W]
#             }



#         sample = self.transform(sample)
#         sample['index'] = idx
#         sample['name'] = left_img_path
#         return sample


class RigidEndoDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.scared_root = self.data_info.get('EVAL_DATA_PATH', self.root)

        # 添加内存缓存参数
        self.load_to_memory = self.data_info.LOAD_TO_MEMORY
        self.data_cache = {}  # 内存缓存

        # 如果启用内存缓存，则预加载所有数据
        if self.load_to_memory:
            self._preload_data_to_memory()

    def _preload_data_to_memory(self):
        """预加载所有数据到内存"""
        print(f"Preloading {len(self.data_list)} samples to memory...")
        for idx in range(len(self.data_list)):
            item = self.data_list[idx]
            if self.mode == 'evaluating':
                full_paths = [os.path.join(self.scared_root, x) for x in item[0:3]]
                left_img_path, right_img_path, disp_img_path = full_paths

                # 读取并缓存数据
                left_img = Image.open(left_img_path).convert('RGB')
                left_img = np.array(left_img, dtype=np.float32)
                right_img = Image.open(right_img_path).convert('RGB')
                right_img = np.array(right_img, dtype=np.float32)
                disp_img = cv2.imread(disp_img_path, cv2.IMREAD_UNCHANGED) / 128
                disp_img = np.array(disp_img, dtype=np.float32)
            else:
                full_paths = [os.path.join(self.root, x) for x in item[0:3]]
                left_img_path, right_img_path, disp_img_path = full_paths

                # 读取并缓存数据
                left_img = Image.open(left_img_path).convert('RGB')
                left_img = np.array(left_img, dtype=np.float32)
                right_img = Image.open(right_img_path).convert('RGB')
                right_img = np.array(right_img, dtype=np.float32)
                disp_img = read_pseudo_disparity(disp_img_path)

            # 存储到缓存
            self.data_cache[idx] = {
                'left': left_img,  # [H, W, 3]
                'right': right_img,  # [H, W, 3]
                'disp': disp_img,  # [H, W]
                'left_img_path': left_img_path
            }

            if idx % 100 == 0:
                print(f"Preloaded {idx+1}/{len(self.data_list)} samples")

        print(f"Preloading complete. Memory usage: {self._get_memory_usage()}")

    def _get_memory_usage(self):
        """估算内存使用量"""
        if not self.data_cache:
            return "0 MB"

        # 计算缓存的总大小
        total_size = 0
        for idx, data in self.data_cache.items():
            for key, value in data.items():
                if key != 'left_img_path':  # 路径字符串很小，忽略不计
                    total_size += value.nbytes

        # 转换为MB或GB
        if total_size > 1024**3:  # 大于1GB
            return f"{total_size / 1024**3:.2f} GB"
        else:
            return f"{total_size / 1024**2:.2f} MB"

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # 如果启用了内存缓存，从缓存中获取数据
        if self.load_to_memory and idx in self.data_cache:
            cached_data = self.data_cache[idx]
            left_img = cached_data['left']
            right_img = cached_data['right']
            disp_img = cached_data['disp']
            left_img_path = cached_data['left_img_path']
        else:
            # 否则从硬盘读取
            if self.mode == 'evaluating':
                full_paths = [os.path.join(self.scared_root, x) for x in item[0:3]]
                left_img_path, right_img_path, disp_img_path = full_paths
                left_img = Image.open(left_img_path).convert('RGB')
                left_img = np.array(left_img, dtype=np.float32)
                right_img = Image.open(right_img_path).convert('RGB')
                right_img = np.array(right_img, dtype=np.float32)
                disp_img = cv2.imread(disp_img_path, cv2.IMREAD_UNCHANGED) / 128
                disp_img = np.array(disp_img, dtype=np.float32)
                assert not np.isnan(disp_img).any(), 'disp_img has nan'
            else:
                full_paths = [os.path.join(self.root, x) for x in item[0:3]]
                left_img_path, right_img_path, disp_img_path = full_paths
                left_img = Image.open(left_img_path).convert('RGB')
                left_img = np.array(left_img, dtype=np.float32)
                right_img = Image.open(right_img_path).convert('RGB')
                right_img = np.array(right_img, dtype=np.float32)
                disp_img = read_pseudo_disparity(disp_img_path)
                assert not np.isnan(disp_img).any(), 'disp_img has nan'

        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
        }

        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = left_img_path
        return sample

def read_pseudo_disparity(path):
    if path.endswith('.npy'):
        return np.load(path).astype(np.float32)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return rgb_to_float(image)


def rgb_to_float(rgb_array):
    r, g, b = rgb_array[..., 0], rgb_array[..., 1], rgb_array[..., 2]

    # 重建uint32（低8位补0）
    uint32_reconstructed = ((r.astype(np.uint32) << 16) |
                           (g.astype(np.uint32) << 8) |
                           b.astype(np.uint32)) << 8

    return uint32_reconstructed.view(np.float32)
