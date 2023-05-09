import h5py
import numpy as np
import torch.utils.data as data
from torchvision.io import read_image
import SimpleITK as sitk
import h5py
import cv2
import os
import json
# 图片路径和标签路径

def nnunet_read():
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


def nnUnet2hdf5():

    
    img_dir1 = "/home/dxm/dxm/Datasets/Ischemic"
    img_dir2 = "/home/dxm/dxm/Datasets/Myocardial"
    
    with open(os.path.join(img_dir1,"dataset.json")) as f:
        annotation1 = json.load(f)["training"]
    image1 = []
    img_labels1 = []
    for i in annotation1:
        image1.append(os.path.join(img_dir1,i['image'].replace("roi","roi_0000") ))
        img_labels1.append(os.path.join(img_dir1,i['label']))
    
    with open(os.path.join(img_dir2,"dataset.json")) as f:
        annotation2 = json.load(f)["training"]
    image2 = []
    img_labels2 = []
    for i in annotation2:
        image2.append(os.path.join(img_dir2,i['image'].replace("roi","roi_0000") ))
        img_labels2.append(os.path.join(img_dir2,i['label']))
    
    del i
    
    save_dir = "/home/dxm/dxm/Datasets/Myocardial+Ischemic_h5py/Myocardial+Ischemic.h5"
    
    # 创建HDF5文件并写入数据
    with h5py.File(save_dir, 'w') as f:
        # 创建数据集
        image_shape = (len(image1) + len(image2), 1, 512, 512,512)
        
        images = f.create_dataset("images", shape=image_shape, dtype=np.float32)
        labels_ds = f.create_dataset("labels", shape=image_shape, dtype=np.float32)
        labels_class = f.create_dataset("labels_class", shape=(len(image1) + len(image2),), dtype=np.int8)
        # 读取图片数据
        for idx in range(len(image1)):
            img_path = image1[idx]
            label_path = img_labels1[idx]
            img = sitk.ReadImage(img_path)
            img = sitk.GetArrayFromImage(img).astype("float32")
            label = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label).astype("float32")
            # breakpoint()
            img_add = 512 - img.shape[0]
            if img_add < 0:
                img_add = -img_add
            label_add = 512 - label.shape[0]
            if label_add < 0:
                label_add = -label_add
            img = np.resize(np.pad(img, ((img_add,0),(0,0), (0,0)), 'edge'), (1,512,512,512))
            label = np.resize(np.pad(label, ((label_add,0), (0,0), (0,0)), 'edge'), (1,512,512,512))
            # breakpoint()
            images[idx] = img
            labels_ds[idx] = label
            labels_class[idx] = 1
            
        for idx in range(len(image2)):
            
            img_path = image2[idx]
            label_path = img_labels2[idx]
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
            images[idx] = img
            labels_ds[idx] = label
            labels_class[idx] = 0
        
        
        # 设置属性
        f.attrs['description'] = 'Myocardial+Ischemic_h5py'
        images.attrs['description'] = 'Myocardial+Ischemic_h5py'
        labels_ds.attrs['description'] = 'Label dataset'

    
if __name__ == "__main__":
    """
    本脚本用来处理从nnunet格式文件转化为hdf5格式文件
    """
    save_dir = "/home/dxm/dxm/Datasets/Myocardial+Ischemic_h5py"
    nnUnet2hdf5()
    
    