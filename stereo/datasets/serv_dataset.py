import os
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate
from stereo.utils.common_utils import get_pos_fullres
import cv2

class SERVDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)


    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item[0:3]]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)
        disp_img = cv2.imread(disp_img_path, cv2.IMREAD_UNCHANGED) / 256
        disp_img = np.array(disp_img, dtype=np.float32)
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
