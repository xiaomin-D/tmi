import os
import json
import torch
import torch.utils.data as data
from torchvision.io import read_image
import SimpleITK as sitk
import numpy as np
import h5py
"""
img_dir = '/home/dxm/dxm/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task100_mycardium'
annotations_file = 'dataset.json'

save_dir = "/home/dxm/dxm/Datasets/Myocardial+Ischemic_h5py/Myocardial+Ischemic.h5"
"""

def build_dataloader(img_dir, batch_size = 8, train = True, transform=None, target_transform=None):
    """
    return dataloader
    """
    loader = data.DataLoader(ct_dataset(img_dir = img_dir, train = train), batch_size=batch_size, shuffle=True,
                    num_workers = 8,pin_memory=False)
    return loader

def build_dataloader_h5(img_dir, batch_size = 8, train = True, transform=None, target_transform=None):
    """
    return dataloader
    """
    loader = data.DataLoader(ct_dataset_h5(img_dir = img_dir, train = train), batch_size=batch_size, shuffle=True,
                    num_workers = 8,pin_memory=False)
    return loader

class ct_dataset(data.Dataset):
    def __init__(self, img_dir, train = True, transform=None, target_transform=None):
        self.img_dir = img_dir
        if train:
            with open(os.path.join(img_dir,"dataset.json")) as f:
                annotation = json.load(f)["training"]
                self.image = []
                self.img_labels = []
                for i in annotation:
                    self.image.append(i["image"].replace("roi","roi_0000") )
                    self.img_labels.append(i["label"])
        else:
            with open(os.path.join(img_dir,"dataset.json")) as f:
                annotation = json.load(f)["test"]
                self.image = annotation
                self.img_labels = []
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image[idx])
        label_path = os.path.join(self.img_dir, self.img_labels[idx])
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img).astype("float32")
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label).astype("float32")
        img_add = 512 - img.shape[0]
        if img_add < 0:
            img_add = -img_add
        label_add = 512 - label.shape[0]
        if label_add < 0:
            label_add = -label_add
        img = np.resize(np.pad(img, ((img_add,0),(0,0), (0,0)), 'edge'), (1,512,512,512))
        label = np.resize(np.pad(label, ((label_add,0), (0,0), (0,0)), 'edge'), (1,512,512,512))
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return img, label
    
class ct_dataset_h5(data.Dataset):
    def __init__(self, img_dir, train = True, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        with h5py.File(self.img_dir, 'r') as f:
            # 假设您想要读取名为'dataset_1'的数据集
            lenth = len(f["images"])
        return lenth

    def __getitem__(self, idx):
        # 打开HDF5文件
        with h5py.File(self.img_dir, 'r') as f:
            # 假设您想要读取名为'dataset_1'的数据集
            images = f["images"]
            labels_ds = f["labels"]
            labels_class = f["labels_class"]
            # 选择数据集的某个区域（从第1到第10行，从第20到第30列）
            img = images[idx] 
            label_d = labels_ds[idx]
            label_class = labels_class[idx]
            # print("dataset description\n  " + f.attrs['description'])
        # images.attrs['description'] = 'Myocardial+Ischemic_h5py'
        # labels_ds.attrs['description'] = 'Label dataset'
        return img, label_d, label_class
    