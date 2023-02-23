import os
import json
import torch
import torch.utils.data as data
from torchvision.io import read_image
import SimpleITK as sitk
import numpy as np
"""
img_dir = '/home/dxm/dxm/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task100_mycardium'
annotations_file = 'dataset.json'
"""

def build_dataloader(img_dir, batch_size = 8, train = True, transform=None, target_transform=None):
    """
    return dataloader
    """
    loader = data.DataLoader(ct_dataset(img_dir = img_dir, train = train), batch_size=batch_size, shuffle=True,
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
        img = sitk.GetArrayFromImage(img).astype(float)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label).astype(float)
        img = np.pad(img, ((512 - img.shape[0],0),(0,0), (0,0)), 'constant')
        label = np.pad(label, ((512 - img.shape[0],0), (0,0), (0,0)), 'constant')
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return img, label
img_dir = '/home/dxm/dxm/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task100_mycardium'
dataloader = build_dataloader(img_dir = img_dir,batch_size=1)
for img, label in dataloader:
    print(img.shape)
    