import SimpleITK as sitk
img_path = '/home/dxm/dxm/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task100_mycardium/imagesTr/43_roi_0000.nii.gz'
img = sitk.ReadImage(img_path)
print(img.GetSize())
img2 = sitk.GetArrayFromImage(img)
print(type(img2))