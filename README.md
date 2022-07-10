## loam_without_ros
loam_without_ros 主要参考大佬 [Tong Qin](http://www.qintonguav.com), [Shaozu Cao](https://github.com/shaozu) 实现的[A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM)进行改写的.改写的目的有两个, 一个是为了学习 loam 算法,另外一个是为了使算法能够脱离 ROS 环境在 MAC 平台上运行及调试. 目前只能在 KITTI 数据集上单线程跑.
<img src="https://github.com/chnhs/blog_image/blob/main/loam_without_ros/kitti.jpg"/>

## 1. MAC 平台下运行前提
### 1.1 Ceres Solver
参考 [Ceres Installation](http://ceres-solver.org/installation.html) 或者在 mac 平台通过 `brew install ceres-solver` 安装.
### 1.2 pcl
参考 [PCL Installation](http://www.pointclouds.org/downloads/linux.html).或者在 mac 平台通过 `brew install pcl` 安装.
### 1.3 Eigen
参考 [Eigen Installation](https://eigen.tuxfamily.org/dox/GettingStarted.html).或者在 mac 平台通过 `brew install eigen` 安装.

## 2. 构建 loam_without_ros
克隆该 github 仓库并在该目录下执行以下代码
```
mkdir build
cd build
cmake ..
make -j
```

## 3. KITTI 例子(Velodyne HDL-64)
下载[KITTI数据集](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)到任意目录下, 然后在命令行执行以下代码
```
PATH_TO_LOAM_PROGRAM --dataset_dir PATH_TO_KITTI_DATASET
```

## 4. 致谢
- 感谢 LOAM(J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time), 激光雷达 slam 的开山系列.
- 感谢[A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM)的开源, 让我能够理解 loam 的思想.
- 感谢[filesystem](https://github.com/gulrak/filesystem), 让我不用升级到 c++17.
- 感谢[cxxopts](https://github.com/jarro2783/cxxopts), 让我不用使用 google 那一套繁重的命令行解析工具.

## 5. 更改项
- 更改了 A-LOAM 中 scan 提取的方式, 采用扫描时$\alpha$角度进行区分.
- 更改了提取边缘点以及平面点的处理, 因为激光雷达是 360°扫描, 所以不在去除前后若干点.
   