## YOLOv9 ncnn



### 1. 修改onnx

```shell
python3 get_ncnn_onnx.py
```

### 2. onnx转ncnn FP32 & FP16

```shell
# 使用ONNX2NCNN生成ncnn支持的模型
onnx2ncnn yolov9-c.onnx yolov9-c.param yolov9-c.bin

# FP16
ncnnoptimize.exe yolov9-c.param yolov9-c.bin yolov9-c-opt.param yolov9-c-opt.bin 65536
```

### 3. onnx转ncnn INT8

+ 1. Optimize model

```shell
./ncnnoptimize yolov9-c.param yolov9-c.bin yolov9-c-opt.param yolov9-c-opt.bin 0
```

+ 2. Create the calibration table file

作者推荐使用验证数据（verification dataset)作为校准集，最好是大于5000张，可以参考ImageNet的样例数据：<https://github.com/nihui/imagenet-sample-images>, 这里我们使用的是xray数据集

```shell
# 构建校准数据列表
find xray/ -type f > imagelist.txt

./ncnn2table yolov9-c-opt.param yolov9-c-opt.bin imagelist.txt yolov9-c.table mean=[0,0,0] norm=[1,1,1] shape=[3,640,640] pixel=RGB thread=2 method=kl
```

+ mean和norm是前处理中`Mat::sunstract_normalize()`的参数，减mean,乘以norm
+ shape是模型input的大小
+ pixel是`Extractor::input()`之前的数据类型
+ thread是CPU的线程数用来并行推理
+ method是PTQ量化方法，支持kl和aciq和easyquant三种量化策略
+ 同时支持多输入的量化

+ 3. Quantize model

```shell
./ncnn2int8 yolov9-c-opt.param yolov9-c-opt.bin yolov9-c-int8.param yolov9-c-int8.bin yolov9-c.table
```

关于ncnn INT8量化的介绍参考

[1].https://zhuanlan.zhihu.com/p/372278785

[2].https://zhuanlan.zhihu.com/p/370689914

[3].https://github.com/Tencent/ncnn/wiki/quantized-int8-inference

### 4. Windows下实现ncnn调用FP32,FP16和INT8的代码





### 5.QT+安卓手机实现手机端部署YOLOv9

