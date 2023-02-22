import torch
import torch.utils.data as data
import os
import json
import pathlib
import SimpleITK as sitk
from torchvision.io import read_image
"""
img_dir = '/home/dxm/dxm/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task100_mycardium'
annotations_file = 'dataset.json'
"""



def build_dataloader():
    """
    return dataloader
    """
    return 0


class ct_dataset(data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        with open(img_dir + "/dataset.json") as f:
           annotation = json.load(f)["train"]
        self.image = []
        self.img_labels = []
        for i in annotation:
            self.image.append(i["image"])
            self.img_labels.append(i["label"])
        print(self.image)
        print(self.img_labels)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image[idx])
        label_path = os.path.join(self.img_dir, self.img_labels[idx])
        img = sitk.ReadImage(img_path)
        
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

    

