

# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers


使用语义分割模型 SegFormer 对driver.mp4驾驶视频进行测试及可视化。

## Installation

在```CUDA 10.1``` and  ```pytorch 1.7.1```设备下，作以下准备工作。 

```
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
cd SegFormer && pip install -e . --user
```

## Visualize

1.执行 python videophoto.py 分割视频 driver.mp4，分割的图片保存在 DRIVER 文件夹里。

2.将 image-demo 里 51-63 行代码注释解除，将 65-68 注释掉。

3.(python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${DEVICE_NAME}] [--palette-thr ${PALETTE})

执行 python demo/image_demo.py "C:/Users/单欣宇/Desktop/SegFormer-master/DRIVER"  local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py\
 segformer.b5.1024x1024.city.160k.pth --device cuda:0 --palette cityscapes
得到DRIVER图片的语义分割结果，保存早Dataset文件夹里。

4.将image_demo里51-63注释掉，65-68注释解除

5.再执行步骤3，即可当得到语义分割后的视频


