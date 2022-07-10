#ifndef LOAM_LASER_ODOMETRY_H_
#define LOAM_LASER_ODOMETRY_H_

#include "Eigen/Eigen"
#include "ceres/loss_function.h"
#include "ceres/manifold.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "pcl/common/transforms.h"
#include "pcl/kdtree/kdtree_flann.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/visualization/cloud_viewer.h"
#include <memory>

namespace loam {
class LaserOdometry {
public:
  /*!
   * 构造函数
   */
  LaserOdometry(pcl::visualization::PCLVisualizer::Ptr visualizer_ptr = nullptr);

  /*!
   * 析构函数
   */
  ~LaserOdometry();

  /*!
   * 处理 scan_registration 获取到的特征点
   * @param corner_point_sharp 边缘点
   * @param corner_point_less_sharp 次边缘点
   * @param surf_point_flat 平面点
   * @param surf_point_less_flat 次平面点
   * @param full_point 全部点
   */
  void processPointClouds(const pcl::PointCloud<pcl::PointXYZI> &corner_point_sharp,
                          const pcl::PointCloud<pcl::PointXYZI> &corner_point_less_sharp,
                          const pcl::PointCloud<pcl::PointXYZI> &surf_point_flat,
                          const pcl::PointCloud<pcl::PointXYZI> &surf_point_less_flat,
                          const pcl::PointCloud<pcl::PointXYZI> &full_point);

private:
  //! 初始化标志位
  bool flag_system_initialized_ = false;

  //! 上一帧的边缘特征点 kdtree 指针, 用于快速查找. 存储的是上一帧的 corner_points_less_sharp
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_corner_last_;

  //! 上一帧的平面特征点kdtree 指针, 用于快速查找. 存储的是上一帧的 surf_points_less_flat
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_surf_last_;

  //! 存储上一帧的 corner_points_less_sharp
  pcl::PointCloud<pcl::PointXYZI> corner_point_last_;

  //! 存储上一帧的 surf_points_less_flat
  pcl::PointCloud<pcl::PointXYZI> surf_point_last_;

  //! 存储上一帧的所有点云
  pcl::PointCloud<pcl::PointXYZI> full_point_last_;

public:
  //! 存储上一帧估计出来的位姿T_W_L及相对位姿 T_last_curr
  Eigen::Quaterniond q_w_l_ = Eigen::Quaterniond(1, 0, 0, 0);
  Eigen::Vector3d t_w_l_ = Eigen::Vector3d(0, 0, 0);
  Eigen::Quaterniond q_last_curr_ = Eigen::Quaterniond(1, 0, 0, 0);
  Eigen::Vector3d t_last_curr_ = Eigen::Vector3d(0, 0, 0);

private:
  //! 最大的迭代次数
  constexpr static int kMaxNumIterations = 2;

  //! 相邻的扫描线判定阈值
  constexpr static int kNearbyScan = 3;

  //! kdtree 搜索距离[m]
  constexpr static int kDistanceSQThreshold = 25;

  //! 一帧激光点云扫描时间[s]
  constexpr static float kScanPeriod = 0.1;

  //! 局部参数化
  std::shared_ptr<ceres::Manifold> eigen_quaternion_manifold_;

  //! lossfunction
  std::shared_ptr<ceres::LossFunction> huber_loss_function_;

  //! pcl 点云显示功能
  pcl::visualization::PCLVisualizer::Ptr visualizer_ptr_;

  /*!
   * 显示激光雷达里程计, 包括特征匹配, 位姿等等
   */
  void showLaserOdometry();
};
} // namespace loam

#endif