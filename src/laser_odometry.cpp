#include "laser_odometry.h"

#include "factors/lidar_edge_factor.h"
#include "factors/lidar_plane_factor.h"
#include <utility>
namespace loam {

LaserOdometry::LaserOdometry(pcl::visualization::PCLVisualizer::Ptr visualizer_ptr)
  : kdtree_corner_last_(new pcl::KdTreeFLANN<pcl::PointXYZI>()),
    kdtree_surf_last_(new pcl::KdTreeFLANN<pcl::PointXYZI>()),
    eigen_quaternion_manifold_(new ceres::EigenQuaternionManifold()),
    huber_loss_function_(new ceres::HuberLoss(0.1)), visualizer_ptr_(std::move(visualizer_ptr))
{
}

LaserOdometry::~LaserOdometry() {}

void LaserOdometry::processPointClouds(
  const pcl::PointCloud<pcl::PointXYZI> &corner_point_sharp,
  const pcl::PointCloud<pcl::PointXYZI> &corner_point_less_sharp,
  const pcl::PointCloud<pcl::PointXYZI> &surf_point_flat,
  const pcl::PointCloud<pcl::PointXYZI> &surf_point_less_flat,
  const pcl::PointCloud<pcl::PointXYZI> &full_point)
{

  // 如果系统没有初始化, 则进行处理
  if (!flag_system_initialized_) {
    flag_system_initialized_ = true;
  } else {
    // 记录边缘点和平面的数量
    size_t corner_point_sharp_size = corner_point_sharp.size();
    size_t surf_point_flat_size = surf_point_flat.size();
    size_t corner_point_last_size = corner_point_last_.size();
    size_t surf_point_last_size = surf_point_last_.size();

    // 迭代次数, 类似于 icp 中的寻找最近点, 然后求解位姿
    for (size_t iter_counter = 0; iter_counter < kMaxNumIterations; iter_counter++) {
      // ceres 问题构造
      ceres::Problem::Options problem_options;
      problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
      problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
      ceres::Problem problem(problem_options);
      problem.AddParameterBlock(q_last_curr_.coeffs().data(), 4, eigen_quaternion_manifold_.get());
      problem.AddParameterBlock(t_last_curr_.data(), 3);

      // 中间使用数据
      pcl::PointXYZI point_after_transform;
      std::vector<int> point_search_index;
      std::vector<float> point_search_square_distance;

      // 边缘点寻找最近邻点并构造优化问题
      for (size_t i = 0; i < corner_point_sharp_size; i++) {

        // 为了方便搜索, 假设当前帧和上一帧的相对位姿相等, 将点转到上一帧的坐标系下
        point_after_transform = corner_point_sharp[i];
        Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3) =
          (q_last_curr_ *
             Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3).cast<double>() +
           t_last_curr_)
            .cast<float>();

        // kdtree 搜索
        kdtree_corner_last_->nearestKSearch(point_after_transform, 1, point_search_index,
                                            point_search_square_distance);

        // 寻找直线所对应的边缘点, 要求 j 和 l 属于不同的 scan
        int edge_point_index_j = -1;
        int edge_point_index_l = -1;
        if (point_search_square_distance[0] < kDistanceSQThreshold) {

          edge_point_index_j = point_search_index[0];
          size_t closest_point_scan_id =
            static_cast<int>(corner_point_last_[edge_point_index_j].intensity);

          float min_point_sq_distance = kDistanceSQThreshold;
          // 增加 scan id, 寻找同一根线的点
          for (int j = edge_point_index_j + 1; j < corner_point_last_size; ++j) {

            size_t curr_point_scan_id = static_cast<int>(corner_point_last_[j].intensity);

            // 同一根扫描线, 不进行处理(正常情况下应该只有等于)
            if (curr_point_scan_id <= closest_point_scan_id) {
              continue;
            }

            // 如果超过了扫描线阈值, 结束搜索
            if (curr_point_scan_id > closest_point_scan_id + kNearbyScan) {
              break;
            }

            // 计算两个点的距离
            float square_distance = (Eigen::Map<Eigen::Vector4f>(corner_point_last_[j].data) -
                                     Eigen::Map<Eigen::Vector4f>(point_after_transform.data))
                                      .head(3)
                                      .norm();

            if (square_distance < min_point_sq_distance) {
              min_point_sq_distance = square_distance;
              edge_point_index_l = j;
            }
          }

          // 减小 scan id, 寻找同一根线的点
          for (int j = edge_point_index_j - 1; j >= 0; --j) {
            size_t curr_point_scan_id = static_cast<int>(corner_point_last_[j].intensity);

            // 同一根扫描线, 不进行处理
            if (curr_point_scan_id <= closest_point_scan_id) {
              continue;
            }

            // 如果超过了扫描线阈值, 结束搜索
            if (curr_point_scan_id < closest_point_scan_id - kNearbyScan) {
              break;
            }

            // 计算两个点的距离
            float square_distance = (Eigen::Map<Eigen::Vector4f>(corner_point_last_[j].data) -
                                     Eigen::Map<Eigen::Vector4f>(point_after_transform.data))
                                      .head(3)
                                      .norm();

            if (square_distance < min_point_sq_distance) {
              min_point_sq_distance = square_distance;
              edge_point_index_l = j;
            }
          }
        }

        // 构造优化问题
        if (edge_point_index_l >= 0) {
          Eigen::Vector3f curr_point =
            Eigen::Map<const Eigen::Vector4f>(corner_point_sharp[i].data).head(3);

          Eigen::Vector3f edge_point_j =
            Eigen::Map<Eigen::Vector4f>(corner_point_last_[edge_point_index_j].data).head(3);

          Eigen::Vector3f edge_point_l =
            Eigen::Map<Eigen::Vector4f>(corner_point_last_[edge_point_index_l].data).head(3);

          ceres::CostFunction *cost_function = LidarEdgeFactor::Create(
            curr_point.cast<double>(), edge_point_j.cast<double>(), edge_point_l.cast<double>());
          problem.AddResidualBlock(cost_function, huber_loss_function_.get(),
                                   q_last_curr_.coeffs().data(), t_last_curr_.data());
        }
      }

      // 平面点寻找最近邻点并构造优化问题
      for (size_t i = 0; i < surf_point_flat_size; i++) {

        // 为了方便搜索, 假设当前帧和上一帧的相对位姿相等, 将点转到上一帧的坐标系下
        point_after_transform = surf_point_flat[i];
        Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3) =
          (q_last_curr_ *
             Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3).cast<double>() +
           t_last_curr_)
            .cast<float>();

        // kdtree 搜索
        kdtree_surf_last_->nearestKSearch(point_after_transform, 1, point_search_index,
                                          point_search_square_distance);

        // 寻找对应平面, 要求 j,l 属于同一条 scan, m 属于不同 scan
        int plane_point_index_j = -1;
        int plane_point_index_l = -1;
        int plane_point_index_m = -1;

        if (point_search_square_distance[0] < kDistanceSQThreshold) {
          // 找到最临近点处于哪个 scan id
          plane_point_index_j = point_search_index[0];
          size_t closest_point_scan_id =
            static_cast<int>(surf_point_last_[plane_point_index_j].intensity);
          float min_point_sq_distance_l = kDistanceSQThreshold;
          float min_point_sq_distance_m = kDistanceSQThreshold;

          // 增加 scan id, 寻找同一平面的点
          for (int j = plane_point_index_j + 1; j < surf_point_last_size; j++) {
            size_t curr_point_scan_id = static_cast<int>(surf_point_last_[j].intensity);

            // 超出搜索线, 进行处理
            if (curr_point_scan_id > (closest_point_scan_id + kNearbyScan)) {
              break;
            }

            // 计算两个点的距离
            float square_distance = (Eigen::Map<Eigen::Vector4f>(surf_point_last_[j].data) -
                                     Eigen::Map<Eigen::Vector4f>(point_after_transform.data))
                                      .head(3)
                                      .norm();

            if (curr_point_scan_id <= closest_point_scan_id &&
                square_distance < min_point_sq_distance_l) {
              min_point_sq_distance_l = square_distance;
              plane_point_index_l = j;
            } else if (curr_point_scan_id > closest_point_scan_id &&
                       square_distance < min_point_sq_distance_m) {
              min_point_sq_distance_m = square_distance;
              plane_point_index_m = j;
            }
          }

          // 减少 scan_id, 寻找同一平面上的点
          for (int j = plane_point_index_j - 1; j >= 0; j--) {
            size_t curr_point_scan_id = static_cast<int>(surf_point_last_[j].intensity);

            // 超出搜索线, 进行处理
            if (curr_point_scan_id < (closest_point_scan_id - kNearbyScan)) {
              break;
            }

            // 计算两个点的距离
            float square_distance = (Eigen::Map<Eigen::Vector4f>(surf_point_last_[j].data) -
                                     Eigen::Map<Eigen::Vector4f>(point_after_transform.data))
                                      .head(3)
                                      .norm();

            if (curr_point_scan_id >= closest_point_scan_id &&
                square_distance < min_point_sq_distance_l) {
              min_point_sq_distance_l = square_distance;
              plane_point_index_l = j;
            } else if (curr_point_scan_id < closest_point_scan_id &&
                       square_distance < min_point_sq_distance_m) {
              min_point_sq_distance_m = square_distance;
              plane_point_index_m = j;
            }
          }
        }

        // 构造优化问题
        if (plane_point_index_l >= 0 && plane_point_index_m >= 0) {
          Eigen::Vector3f curr_point =
            Eigen::Map<const Eigen::Vector4f>(surf_point_flat[i].data).head(3);

          Eigen::Vector3f plane_point_j =
            Eigen::Map<Eigen::Vector4f>(surf_point_last_[plane_point_index_j].data).head(3);

          Eigen::Vector3f plane_point_l =
            Eigen::Map<Eigen::Vector4f>(surf_point_last_[plane_point_index_l].data).head(3);

          Eigen::Vector3f plane_point_m =
            Eigen::Map<Eigen::Vector4f>(surf_point_last_[plane_point_index_m].data).head(3);

          ceres::CostFunction *cost_function =
            LidarPlaneFactor::Create(curr_point.cast<double>(), plane_point_j.cast<double>(),
                                     plane_point_l.cast<double>(), plane_point_m.cast<double>());
          problem.AddResidualBlock(cost_function, huber_loss_function_.get(),
                                   q_last_curr_.coeffs().data(), t_last_curr_.data());
        }
      }

      // 求解优化问题
      ceres::Solver::Options solver_options;
      solver_options.linear_solver_type = ceres::DENSE_QR;
      solver_options.max_num_iterations = 4;
      solver_options.minimizer_progress_to_stdout = false;
      ceres::Solver::Summary summary;
      ceres::Solve(solver_options, &problem, &summary);
    }

    // 更新位姿
    t_w_l_ = t_w_l_ + q_w_l_ * t_last_curr_;
    q_w_l_ = q_w_l_ * q_last_curr_;
  }

  // 保存数据
  corner_point_last_ = corner_point_less_sharp;
  surf_point_last_ = surf_point_less_flat;
  full_point_last_ = full_point;

  // 更新 kdtree
  kdtree_corner_last_->setInputCloud(corner_point_last_.makeShared());
  kdtree_surf_last_->setInputCloud(surf_point_last_.makeShared());

  // 展示 laser odometry
  // showLaserOdometry();
}

void LaserOdometry::showLaserOdometry()
{
  static int laser_odometry_num = 0;
  if (visualizer_ptr_ == nullptr) {
    return;
  }

  Eigen::Affine3f affine = Eigen::Affine3f::Identity();
  affine.translate(t_w_l_.cast<float>());
  affine.rotate(q_w_l_.toRotationMatrix().cast<float>());

  pcl::transformPointCloud(full_point_last_, full_point_last_, affine);

  auto full_point_last_ptr = full_point_last_.makeShared();
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> gray(full_point_last_ptr, 192,
                                                                        192,
                                                                        192); // rgb
  visualizer_ptr_->addPointCloud<pcl::PointXYZI>(full_point_last_ptr, gray,
                                                 "points" + std::to_string(laser_odometry_num++));

  visualizer_ptr_->addCoordinateSystem(1, affine);
}

} // namespace loam