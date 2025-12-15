import os

from PIL import Image
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from commons import RGB2YCrCb
import torch
import torch.utils.data as Data
import h5py
import numpy as np
to_tensor = transforms.Compose([transforms.ToTensor()])


class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        return torch.Tensor(VIS), torch.Tensor(IR)

to_tensor = transforms.Compose([transforms.ToTensor()])

class Downsample(nn.Module):
    def __init__(self, scale_factor):
        super(Downsample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        return self.pool(x)

class MSRS_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得TNO数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'MRI':
                self.inf_path = temp_path  # 获得红外路径
            else:
                self.vis_path = temp_path  # 获得可见光路径
            # if sub_dir == 'MRI':
            #     self.inf_path = temp_path
            # elif sub_dir == 'SPECT':
            #     self.vis_path = temp_path
        self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')  # 获取红外图像
        vis_image = Image.open(os.path.join(self.vis_path, name)).convert('RGB')
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name
        # return vis_image, inf_image

    def __len__(self):
        return len(self.name_list)
