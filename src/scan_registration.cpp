#include "scan_registration.h"

#include "Eigen/Eigen"
#include "utils/utils.h"
#include <utility>
namespace loam {

ScanRegistration::ScanRegistration(pcl::visualization::PCLVisualizer::Ptr visualizer_ptr)
  : visualizer_ptr_(std::move(visualizer_ptr))
{
}

ScanRegistration::~ScanRegistration() {}

void ScanRegistration::processPointClouds(const pcl::PointCloud<pcl::PointXYZI> &input_point_cloud)
{
  // 每帧数据都需要清理一遍, 避免出现错误
  corner_points_sharp_.clear();
  corner_points_less_sharp_.clear();
  surf_points_flat_.clear();
  surf_points_less_flat_.clear();

  // 去除无穷远点及距离过近的点
  pcl::PointCloud<pcl::PointXYZI> point_cloud;
  removeInfiniteAndClosedPoint(input_point_cloud, point_cloud, kMinSquareRangeForPoint);

  // 对点按照扫描线束进行分类, 同时计算每个点的扫描时间(存入 intensity)
  std::vector<pcl::PointCloud<pcl::PointXYZI>> scans;
  splitPointCloud(point_cloud, scans);

  // 对每个线束进行处理, 提取特征
  for (const auto &scan : scans) {
    extractFeatureFromScan(scan);
  }

  // 展示结果
  showScanRegistration();
}

void ScanRegistration::removeInfiniteAndClosedPoint(const pcl::PointCloud<pcl::PointXYZI> &input,
                                                    pcl::PointCloud<pcl::PointXYZI> &output,
                                                    float threshold_of_distance)
{
  // 如果另存的话, 需要提前设置拷贝和分配内存
  if (&input != &output) {
    output.header = input.header;
    output.points.resize(input.points.size());
  }
  // 移除无穷远点
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(input, output, indices);

  // 移除过近点
  float square_threshold = threshold_of_distance * threshold_of_distance;
  size_t j = 0;
  for (size_t i = 0; i < output.size(); ++i) {
    if (utils::pcl::DistanceSquare(output[i]) < square_threshold) {
      continue;
    }
    output.points[j] = output.points[i];
    j++;
  }

  // 点云参数重置
  if (j != output.points.size()) {
    output.points.resize(j);
  }
  output.height = 1;
  output.width = static_cast<uint32_t>(j);
  output.is_dense = true;
}

void ScanRegistration::splitPointCloud(const pcl::PointCloud<pcl::PointXYZI> &input_point_cloud,
                                       std::vector<pcl::PointCloud<pcl::PointXYZI>> &output_scans)
{

  // 防止出错, 提前清除
  output_scans.clear();
  output_scans.resize(kScanNum);

  // 根据扫描的三维点计算每个点属于哪个 scan
  int scan_index = 0;
  float last_omega = -1;
  float last_alpha = -1;
  for (size_t i = 0; i < input_point_cloud.size(); ++i) {

    // 计算两个角度
    const pcl::PointXYZI &point = input_point_cloud.points[i];
    float omega =
      static_cast<float>(atan(point.z / sqrt(pow(point.x, 2) + pow(point.y, 2))) * 180 / M_PI);
    float alpha = utils::angle::convertAngleTo2Pi(atan2(point.y, point.x));
    float relative_time = static_cast<float>(alpha / (2 * M_PI));

    // 初始化
    if (last_alpha < 0 && last_omega < 0) {
      last_omega = omega;
      last_alpha = alpha;
    }

    // 判断是否需要增加 index
    if ((alpha - last_alpha) < -1) {
      scan_index++;
    }

    last_omega = omega;
    last_alpha = alpha;

    // 只取[0,50)的点云数据
    if (scan_index > 50) {
      break;
    }

    // 计算扫描角度
    output_scans[scan_index].push_back(pcl::PointXYZI(
      point.x, point.y, point.z,
      static_cast<float>(kScanPeriod * relative_time + static_cast<float>(scan_index))));
  }
} // namespace loam

void ScanRegistration::extractFeatureFromScan(const pcl::PointCloud<pcl::PointXYZI> &scan)
{

  if (scan.empty()) {
    return;
  }

  int scan_size = scan.size();

  // 计算scan 中每个点的曲率
  std::vector<float> cloud_curvature(scan.size(), -1.0f);
  std::vector<int> cloud_index_for_sort(scan_size, 1);
  Eigen::Vector3f point_sum(0, 0, 0);

  for (int i = -kPointNumForFeatureExtractHalf; i <= kPointNumForFeatureExtractHalf; ++i) {

    int index = (i + scan_size) % scan_size;
    point_sum += Eigen::Map<const Eigen::Vector4f>(scan.points[index].data).head(3);
  }

  for (size_t i = 0; i < scan_size; i++) {

    // 曲率计算公式
    Eigen::Vector3f diff =
      point_sum - 11 * Eigen::Map<const Eigen::Vector4f>(scan.points[i].data).head(3);

    cloud_curvature[i] =
      diff.norm() / Eigen::Map<const Eigen::Vector4f>(scan.points[i].data).head(3).norm();

    // 去头部, 增尾部
    size_t head_index = (i - kPointNumForFeatureExtractHalf + scan_size) % scan_size;
    size_t tail_index = (i + 1 + kPointNumForFeatureExtractHalf + scan_size) % scan_size;

    point_sum -= Eigen::Map<const Eigen::Vector4f>(scan.points[head_index].data).head(3);
    point_sum += Eigen::Map<const Eigen::Vector4f>(scan.points[tail_index].data).head(3);

    // index 赋值, 用于后续按曲率排序
    cloud_index_for_sort[i] = i;
  }

  // 将 scan 划分为若干个区域, 保证特征分布均匀
  std::vector<bool> cloud_neighbor_picked(scan_size, false);
  std::vector<FeatureType> cloud_point_type(scan_size, kFeatureUnknow);
  for (int i = 0; i < kSplitNumForFeatureExtract; ++i) {

    // 获取分段的起始和终点
    int start_index = scan_size * i / kSplitNumForFeatureExtract;
    int end_index = scan_size * (i + 1) / kSplitNumForFeatureExtract - 1;

    // 对区间范围内的点按曲率进行排序
    std::sort(cloud_index_for_sort.begin() + start_index,
              cloud_index_for_sort.begin() + end_index + 1,
              [&](const int a, const int b) { return cloud_curvature[a] < cloud_curvature[b]; });

    // 寻找边缘特征点(corner points sharp/less sharp)
    int corner_point_num = 0;
    for (int j = end_index; j >= start_index; j--) {
      int index = cloud_index_for_sort[j];
      // 如果满足大于曲率且不聚集的条件, 则进行处理
      if (!cloud_neighbor_picked[index] && cloud_curvature[index] > kCurvatureThreshold) {
        // 曲率满足条件且数量也满足, 则设置类型并放入
        corner_point_num++;
        if (corner_point_num <= kSharpPointNumPerSplit) {
          cloud_point_type[index] = kFeatureCornerSharp;
          corner_points_sharp_.push_back(scan.points[index]);
          corner_points_less_sharp_.push_back(scan.points[index]);
        } else if (corner_point_num <= kLessSharpPointNumPerSplit) {
          cloud_point_type[index] = kFeatureCornerLessSharp;
          corner_points_less_sharp_.push_back(scan.points[index]);
        } else {
          break;
        }

        // 防止特征点聚集, 进行处理
        cloud_neighbor_picked[index] = true;
        for (int k = 1; k <= kPointNumForFeatureExtractHalf; k++) {
          double distance = (Eigen::Map<const Eigen::Vector4f>(
                               scan.points[(index + k + scan_size) % scan_size].data) -
                             Eigen::Map<const Eigen::Vector4f>(
                               scan.points[(index + k - 1 + scan_size) % scan_size].data))
                              .head(3)
                              .norm();
          if (distance > kDistanceForFeatureSuppression) {
            break;
          };

          cloud_neighbor_picked[(index + k + scan_size) % scan_size] = true;
        }
        for (int k = 1; k <= kPointNumForFeatureExtractHalf; k++) {
          double distance = (Eigen::Map<const Eigen::Vector4f>(
                               scan.points[(index - k + scan_size) % scan_size].data) -
                             Eigen::Map<const Eigen::Vector4f>(
                               scan.points[(index - k + 1 + scan_size) % scan_size].data))
                              .head(3)
                              .norm();
          if (distance > kDistanceForFeatureSuppression) {
            break;
          };

          cloud_neighbor_picked[(index - k + scan_size) % scan_size] = true;
        }
      }
    }

    // 寻找平面特征点(surf points)
    int surf_point_num = 0;
    for (int j = start_index; j <= end_index; j++) {
      int index = cloud_index_for_sort[j];

      if (!cloud_neighbor_picked[index] && cloud_curvature[index] <= kCurvatureThreshold) {

        // 确认为平面点
        cloud_point_type[index] = kFeatureSurfFlat;
        surf_points_flat_.push_back(scan.points[index]);

        surf_point_num++;

        if (surf_point_num >= kFlatPointNumPerSplit) {
          break;
        }

        // 防止特征聚集
        cloud_neighbor_picked[index] = true;
        for (int k = 1; k <= kPointNumForFeatureExtractHalf; k++) {
          double distance = (Eigen::Map<const Eigen::Vector4f>(
                               scan.points[(index + k + scan_size) % scan_size].data) -
                             Eigen::Map<const Eigen::Vector4f>(
                               scan.points[(index + k - 1 + scan_size) % scan_size].data))
                              .head(3)
                              .norm();
          if (distance > kDistanceForFeatureSuppression) {
            break;
          };

          cloud_neighbor_picked[(index + k + scan_size) % scan_size] = true;
        }
        for (int k = 1; k <= kPointNumForFeatureExtractHalf; k++) {
          double distance = (Eigen::Map<const Eigen::Vector4f>(
                               scan.points[(index - k + scan_size) % scan_size].data) -
                             Eigen::Map<const Eigen::Vector4f>(
                               scan.points[(index - k + 1 + scan_size) % scan_size].data))
                              .head(3)
                              .norm();
          if (distance > kDistanceForFeatureSuppression) {
            break;
          };

          cloud_neighbor_picked[(index - k + scan_size) % scan_size] = true;
        }
      }
    }

    // 剩余的为次平面特征点(surf points less flat)
    pcl::PointCloud<pcl::PointXYZI>::Ptr surf_points_less_flat_tmp(
      new pcl::PointCloud<pcl::PointXYZI>());
    for (int j = start_index; j < end_index; j++) {
      if (cloud_point_type[j] == kFeatureUnknow || cloud_point_type[j] == kFeatureSurfFlat) {
        surf_points_less_flat_tmp->push_back(scan.points[j]);
      }
    }

    // 此平面特征点过于密集, 需要进行体素滤波
    pcl::VoxelGrid<pcl::PointXYZI> down_size_filter;
    down_size_filter.setInputCloud(surf_points_less_flat_tmp);
    down_size_filter.setLeafSize(kVoxelGridX, kVoxelGridY, kVoxelGridZ);
    down_size_filter.filter(*surf_points_less_flat_tmp);
    surf_points_less_flat_ += *surf_points_less_flat_tmp;
  }
}

void ScanRegistration::showScanRegistration()
{
  if (visualizer_ptr_ == nullptr) {
    return;
  }
}

} // namespace loam