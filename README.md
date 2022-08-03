# E-HANet:
受基于标准相机的光流估计算法[RAFT](https://github.com/princeton-vl/RAFT)和注意力机制启发，提出提出基于事件的混合注意力网络E-HANet (Event-based Hybrid Attention Network)用于稠密光流估计

![光流估计结果](./vis/result.png)

## 算法总体网络框架
将时间窗口上相邻的两组事件表示为正负堆叠事件片分别输入权重共享的SE-特征提取器，输出𝑓_(𝑖−1) 和𝑓_𝑐𝑜𝑛𝑡𝑒𝑥𝑡，两者用于计算相关性矩阵和运动特征。混合注意力加权模块分别对事件特征𝑓_𝑐𝑜𝑛𝑡𝑒𝑥𝑡和运动特征𝑓_𝑚𝑜𝑡𝑖𝑜𝑛计算空间注意力和通道注意力，实现运动特征的全局聚合。最后使用GRU模型对光流进行迭代更新。

![光流估计结果](./vis/networks.png)

## 原型系统实现
通过Java程序调用Python程序，实现基于JavaWeb的光流估计原型系统，将用户上传的事件流数据输入网络模型，完成光流估计，对事件图像和光流图像进行可视化，并返回Numpy格式的光流估计结果供用户下载

#### 首页：用户上传事件流数据(.h5格式)
![首页](./Java/vis/index.jpg)

#### 展示：将生成的事件图像和光流估计结果可视化，并返回给用户
![展示](./Java/vis/show.jpg)

#### 下载：将光流保存为.npy格式供用户下载
![下载](./Java/vis/download.jpg)

## Datasets

### [DSEC](https://dsec.ifi.uzh.ch/)

### [MVSEC](https://daniilidis-group.github.io/mvsec/)
