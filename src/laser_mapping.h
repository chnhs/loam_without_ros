#ifndef LOAM_LASER_MAPPING_H_
#define LOAM_LASER_MAPPING_H_

#include "Eigen/Eigen"
#include "ceres/loss_function.h"
#include "ceres/manifold.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/kdtree/kdtree_flann.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/visualization/cloud_viewer.h"
#include <array>

namespace loam {
class LaserMapping {

public:
  /*!
   * 构造函数
   */
  LaserMapping(pcl::visualization::PCLVisualizer::Ptr visualizer_ptr = nullptr);

  /*!
   * 析构函数
   */
  ~LaserMapping();

  /*!
   * 处理点云数据
   * @param corner_point_less_sharp 次边缘点数据
   * @param surf_point_less_flat 次平面点数据
   * @param full_point 全部点
   * @param q_wodom_l 当前帧激光雷达在 odometry 坐标系下的姿态
   * @param t_wodom_l 当前帧激光雷达在 odometry 坐标系下的位置
   */
  void processPointClouds(const pcl::PointCloud<pcl::PointXYZI> &corner_point_less_sharp,
                          const pcl::PointCloud<pcl::PointXYZI> &surf_point_less_flat,
                          const pcl::PointCloud<pcl::PointXYZI> &full_point,
                          const Eigen::Quaterniond &q_wodom_l, const Eigen::Vector3d &t_wodom_l);

private:
  /*!
   * 调整 cube, 使得当前激光雷达的位置所处的 subcube 不在整个大的 cube 边缘
   * 同时构建特征地图, 用于优化位姿
   * @note 更改类变量zero_in_cube_width_,zero_in_cube_height_,zero_in_cube_depth_
   * @note 更改类变量 corner_map_, surf_map_
   */
  void adjustCubeAndConstructMap();

  /*!
   * 优化位姿并将该帧点云根据优化后的位姿存入到 cube 中
   * @param corner_point_less_sharp 次边缘点数据
   * @param surf_point_less_flat 次平面点数据
   * @note 更改类变量q_w_l_, t_w_l_
   * @note 更改类变量subcube_corner_array_, subcube_surf_array_
   */
  void optimizePoseAndPushCube(const pcl::PointCloud<pcl::PointXYZI> &corner_point_less_sharp,
                               const pcl::PointCloud<pcl::PointXYZI> &surf_point_less_flat);

  //! 立方体参数, 将点云表示成一个大的立方体(Cube)表示,同时将立方体分割成若干个子立方体(SubCube)
  constexpr static int kCubeWidthNum = 21;
  constexpr static int kCubeHeightNum = 21;
  constexpr static int kCubeDepthNum = 11;
  constexpr static float kSubCubeSideLength = 50.f;
  constexpr static float kSubCubeSideLengthHalf = kSubCubeSideLength / 2.f;
  constexpr static int kSubCubeNum = kCubeWidthNum * kCubeHeightNum * kCubeDepthNum;

  /*!
   * 根据 subcube 在整个 cube 中的坐标计算 index
   * @param i,j,k 三维坐标
   */
  inline int getArrayIndex(int i, int j, int k)
  {
    return i + kCubeWidthNum * j + kCubeWidthNum * kCubeHeightNum * k;
  }

  //! 边缘及平面特征地图
  pcl::PointCloud<pcl::PointXYZI> corner_map_;
  pcl::PointCloud<pcl::PointXYZI> surf_map_;

  //! kdtree 指针, 用于地图快速搜索
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_corner_map_;
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_surf_map_;

  //! 立方体数组, 每个立方体中存储这部分点云
  std::array<pcl::PointCloud<pcl::PointXYZI>::Ptr, kSubCubeNum> subcube_corner_array_;
  std::array<pcl::PointCloud<pcl::PointXYZI>::Ptr, kSubCubeNum> subcube_surf_array_;

  //! 待估计参数, 当前帧激光雷达在全局坐标系下的位姿
  Eigen::Quaterniond q_wmap_l_ = Eigen::Quaterniond(1, 0, 0, 0);
  Eigen::Vector3d t_wmap_l_ = Eigen::Vector3d(0, 0, 0);

  //! 认为laser odometry 估计出来的位姿是有偏差的, 且偏差都在 w 系上,b 系是对齐的
  Eigen::Quaterniond q_wmap_wodom_ = Eigen::Quaterniond(1, 0, 0, 0);
  Eigen::Vector3d t_wmap_wodom_ = Eigen::Vector3d(0, 0, 0);

  //! 体素滤波, 用于缩减点云数量
  constexpr static double kVoxelGridCornerResolution = 0.4;
  constexpr static double kVoxelGridSurfResolution = 0.8;
  pcl::VoxelGrid<pcl::PointXYZI> downsize_filter_corner_;
  pcl::VoxelGrid<pcl::PointXYZI> downsize_filter_surf_;

  //! [0,0,0]坐标 在 cube 中的 index
  int zero_in_cube_width_ = 10;
  int zero_in_cube_height_ = 10;
  int zero_in_cube_depth_ = 5;

  //! 位姿求解参数
  constexpr static int kMaxNumIterations = 2;

  //! 局部参数化
  std::shared_ptr<ceres::Manifold> eigen_quaternion_manifold_;

  //! lossfunction
  std::shared_ptr<ceres::LossFunction> huber_loss_function_;

  //! pcl 点云显示功能
  pcl::visualization::PCLVisualizer::Ptr visualizer_ptr_;

  /*!
   *  显示激光雷达建图, 包括地图, 位姿等等
   */
  void showLaserMapping();
};
} // namespace loam

#endif