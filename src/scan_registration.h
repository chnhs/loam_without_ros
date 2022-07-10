#ifndef LOAM_SCAN_REGISTRATION_H_
#define LOAM_SCAN_REGISTRATION_H_

#include "pcl/filters/filter.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/kdtree/kdtree_flann.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/visualization/cloud_viewer.h"
#include "utils/utils.h"

namespace loam {
class ScanRegistration {
public:
  /*!
   * 构造函数
   */
  ScanRegistration(pcl::visualization::PCLVisualizer::Ptr visualizer_ptr = nullptr);

  /*!
   * 析构函数
   */
  ~ScanRegistration();

  /*!
   * 处理当前帧点云数据
   * @param input_point_cloud 输入的点云数据
   */
  void processPointClouds(const pcl::PointCloud<pcl::PointXYZI> &input_point_cloud);

private:
  enum FeatureType {
    kFeatureCornerSharp = 0,
    kFeatureCornerLessSharp,
    kFeatureSurfFlat,
    kFeatureSurfLessFlat,
    kFeatureUnknow
  };

  /*!
   * 移除无穷远及过近的点云数据
   * @param input 输入点云数据
   * @param output 输出点云数据
   * @param threshold_of_distance 判断点云距离过近的阈值
   */
  void removeInfiniteAndClosedPoint(const pcl::PointCloud<pcl::PointXYZI> &input_point_cloud,
                                    pcl::PointCloud<pcl::PointXYZI> &output_point_cloud,
                                    float threshold_of_distance);

  /*!
   * 将点云按照扫描线进行分割, 用于特征提取; 同时计算每个扫描点的时间
   * @param input_point_cloud 输入的点云数据
   * @param output_scans 输出多个点云数据,每个点云数据代表一条扫描线
   */
  void splitPointCloud(const pcl::PointCloud<pcl::PointXYZI> &input_point_cloud,
                       std::vector<pcl::PointCloud<pcl::PointXYZI>> &output_scans);

  /*!
   * 显示分割出来的scan, 提取出来的特征
   */
  void showScanRegistration();

  /*!
   * 对每个 scan 提取特征
   * @param scan 每个 scan 的点云集合
   */
  void extractFeatureFromScan(const pcl::PointCloud<pcl::PointXYZI> &scan);

  //! 每个点距离目标的最小距离的平方, 单位 m
  static constexpr float kMinSquareRangeForPoint = 0.1;

  //! 扫描周期, 单位 s
  static constexpr float kScanPeriod = 0.1;

  //! 激光雷达扫描线束数量
  static constexpr int kScanNum = 64;

  //! 特征点提取所需激光点云数量
  static constexpr int kPointNumForFeatureExtractHalf = 5;

  //! 特征点提取所分割的区域数量
  static constexpr int kSplitNumForFeatureExtract = 6;

  //! 特征点提取每个区域提取的特征数量
  static constexpr int kSharpPointNumPerSplit = 2;
  static constexpr int kLessSharpPointNumPerSplit = 20;
  static constexpr int kFlatPointNumPerSplit = 4;

  //! 特征点提取曲率阈值
  static constexpr float kCurvatureThreshold = 0.1;

  //! 特征点提取相对距离, 防止特征点提取聚集, 有点类似于非极大值抑制
  static constexpr float kDistanceForFeatureSuppression = 0.05;

  //! 体素栅格滤波的长宽高, 单位 m
  static constexpr float kVoxelGridX = 0.2;
  static constexpr float kVoxelGridY = 0.2;
  static constexpr float kVoxelGridZ = 0.2;

public:
  //! 存储每一帧提取出来的点云特征
  pcl::PointCloud<pcl::PointXYZI> corner_points_sharp_;
  pcl::PointCloud<pcl::PointXYZI> corner_points_less_sharp_;
  pcl::PointCloud<pcl::PointXYZI> surf_points_flat_;
  pcl::PointCloud<pcl::PointXYZI> surf_points_less_flat_;

private:
  //! pcl 点云显示功能
  pcl::visualization::PCLVisualizer::Ptr visualizer_ptr_;
};

} // namespace loam
#endif