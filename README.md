# argoverse-centerpoint-simplified

## Installation

I used Python 3.8.3

**Dependencies**
- SparseConv (build locally)
- DCN (build locally), produce deform_conv_cuda.cpython-38-x86_64-linux-gnu.so
- Iou3dNMS (build locally), produce iou3d_nms_cuda.cpython-38-x86_64-linux-gnu.so
- Pytorch 1.7.1 (check with `python -c "import torch; print(torch.__version__)"`)
- argoverse-api
- CUDA 11.0 (check with `python -c "import torch; print(torch.version.cuda)"`)

If you wish to run `viz_aggregated_sweeps.py`, you must run:
Mayavi Environment: https://github.com/mne-tools/mne-python/blob/master/environment.yml

## Bug Fixed addressed

https://github.com/pytorch/pytorch/issues/29642

use torch::RegisterOperators

Not a problem if you use latest Pytorch?


nvcc fatal   : Unknown option '-Wall'
https://github.com/traveller59/spconv/issues/69
CUDACXX=/usr/local/cuda/bin/nvcc python setup.py bdist_wheel
pip install * --force-reinstall


https://pytorch.org/get-started/previous-versions/


- RuntimeError: /nethome/jlambert30/spconv/src/spconv/indice.cu 274
cuda execution failed with error 98 invalid device function
prepareSubMGridKernel failed
https://github.com/traveller59/spconv/issues/34
Make sure you use the same CUDA version for all installations (set CUDA_HOME before building anything)

## Deformable Convolution 

Added here:
https://github.com/pytorch/vision/pull/1586/files

## nuScenes Coordinate System

The nuScenes egovehicle coordinate frame is situated on the ground underneath the center of rear axle. Consider the pose of the LiDAR sensor in the egovehicle frame:
```python
egovehicle_SE3_lidar.translation
array([0.94, 0.      , 1.84 ])
```
This means the LiDAR is on the center of the car, 1.84 meters above the ground. It is also almost one meter forward (+x) from the rear axle.

What about the relative orientation between the frames?
```python
Rotation.from_matrix(egovehicle_SE3_lidar.rotation).as_euler('zyx', degrees=True)
array([-89.9,   1.4,   0.3])
```
We see that the LiDAR frame is basically rotated -90 degrees from the egovehicle frame

```python
np.round(egovehicle_SE3_lidar.transform_point_cloud(np.eye(3)),2)
array([[ 0.95, -1.  ,  1.83],
       [ 1.94,  0.  ,  1.82],
       [ 0.97, -0.01,  2.84]])
```
<p align="center">
  <img src="https://www.nuscenes.org/public/images/data.png" height="400">
  <img src="https://user-images.githubusercontent.com/16724970/102704589-60dcde00-424b-11eb-8997-aff36f705404.jpg" height="400">
</p>

## Another Option for DCN:
Pytorch deformable conv: https://pytorch.org/docs/stable/_modules/torchvision/ops/deform_conv.html#deform_conv2d

