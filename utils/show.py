"""
Show the image
"""
import SimpleITK as sitk
import vtk
heart_image = sitk.ReadImage('/data2/dxm/Task300_ischemic/imagesTr/45_roi_0000.nii.gz')
vessel_image = sitk.ReadImage('/data2/dxm/Task300_ischemic/labelsTr/45_roi.nii.gz')
# 转换类型为float32
heart_image_float = sitk.Cast(heart_image, sitk.sitkFloat32)
vessel_image_float = sitk.Cast(vessel_image, sitk.sitkFloat32)

# 叠加图像
overlay_image_float = heart_image_float + vessel_image_float

# 将图像转换为uint8类型
overlay_image = sitk.Cast(sitk.RescaleIntensity(overlay_image_float), sitk.sitkUInt8)





# 将SimpleITK图像转换为VTK图像数据
vtk_data = sitk.GetArrayViewFromImage(overlay_image)
vtk_data = vtk_data[::-1,:,:].copy()  # 因为VTK中的坐标系与SimpleITK不同，所 以需要调整轴的顺序
overlay_image_vtk = vtk.vtkImageData()
overlay_image_vtk.SetDimensions(overlay_image.GetSize()[::-1])
overlay_image_vtk.SetOrigin(overlay_image.GetOrigin()[::-1])
overlay_image_vtk.SetSpacing(overlay_image.GetSpacing()[::-1])
overlay_image_vtk.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
overlay_image_vtk.GetPointData().SetScalars(vtk_data)

# 创建渲染器和渲染窗口
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# 使用vtkFixedPointVolumeRayCastMapper进行体绘制
volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
volume_mapper.SetInputData(overlay_image_vtk)
volume_mapper.SetBlendModeToComposite()
volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)

# 设置心肌半透明
volume_property = volume.GetProperty()
volume_property.SetColor(vtk.vtkColorTransferFunction())
volume_property.SetScalarOpacity(vtk.vtkPiecewiseFunction())
volume_property.ShadeOff()
volume_property.SetOpacity(0.5)

# 将体绘制添加到渲染器
renderer.AddVolume(volume)

# 设置背景颜色和相机位置
renderer.SetBackground(1, 1, 1)
renderer.ResetCamera()

# 启动交互式窗口
render_window.Render()
render_window_interactor.Start()

