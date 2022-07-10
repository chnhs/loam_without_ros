#include "laser_mapping.h"

#include "factors/lidar_edge_factor.h"
#include "factors/lidar_plane_factor.h"
#include "factors/lidar_plane_norm_factor.h"
#include <utility>
namespace loam {

LaserMapping::LaserMapping(pcl::visualization::PCLVisualizer::Ptr visualizer_ptr)
  : kdtree_corner_map_(new pcl::KdTreeFLANN<pcl::PointXYZI>()),
    kdtree_surf_map_(new pcl::KdTreeFLANN<pcl::PointXYZI>()),
    eigen_quaternion_manifold_(new ceres::EigenQuaternionManifold()),
    huber_loss_function_(new ceres::HuberLoss(0.1)), visualizer_ptr_(std::move(visualizer_ptr))
{
  // 初始化subcube
  for (int i = 0; i < kSubCubeNum; ++i) {
    subcube_corner_array_[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
    subcube_surf_array_[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
  }
  // 初始化voxel filter
  downsize_filter_corner_.setLeafSize(kVoxelGridCornerResolution, kVoxelGridCornerResolution,
                                      kVoxelGridCornerResolution);
  downsize_filter_surf_.setLeafSize(kVoxelGridSurfResolution, kVoxelGridSurfResolution,
                                    kVoxelGridSurfResolution);
}

LaserMapping::~LaserMapping() {}

void LaserMapping::processPointClouds(
  const pcl::PointCloud<pcl::PointXYZI> &corner_point_less_sharp,
  const pcl::PointCloud<pcl::PointXYZI> &surf_point_less_flat,
  const pcl::PointCloud<pcl::PointXYZI> &full_point, const Eigen::Quaterniond &q_wodom_l,
  const Eigen::Vector3d &t_wodom_l)
{

  // 计算当前激光雷达在 wmap 坐标系下的位姿
  q_wmap_l_ = q_wmap_wodom_ * q_wodom_l;
  t_wmap_l_ = t_wmap_wodom_ + q_wmap_wodom_ * t_wodom_l;

  // 根据当前激光雷达的位置, 调整 cube; 构建地图用于优化位姿
  adjustCubeAndConstructMap();

  // 优化位姿并将当前帧点云数据存入到 cube 中
  optimizePoseAndPushCube(corner_point_less_sharp, surf_point_less_flat);

  // 更新 w 系下的相对位姿
  q_wmap_wodom_ = q_wmap_l_ * q_wodom_l.inverse();
  t_wmap_wodom_ = t_wmap_l_ - q_wmap_wodom_ * t_wodom_l;

  // 展示 laser mapping 结果
  showLaserMapping();
}

void LaserMapping::adjustCubeAndConstructMap()
{
  // 当前激光雷达位置处于哪个 subcube 中, 获取其坐标
  int subcube_index_i =
    int((t_wmap_l_.x() + kSubCubeSideLengthHalf) / kSubCubeSideLength) + zero_in_cube_width_;
  int subcube_index_j =
    int((t_wmap_l_.y() + kSubCubeSideLengthHalf) / kSubCubeSideLength) + zero_in_cube_height_;
  int subcube_index_k =
    int((t_wmap_l_.z() + kSubCubeSideLengthHalf) / kSubCubeSideLength) + zero_in_cube_depth_;

  if (t_wmap_l_.x() + kSubCubeSideLengthHalf < 0)
    subcube_index_i--;
  if (t_wmap_l_.y() + kSubCubeSideLengthHalf < 0)
    subcube_index_j--;
  if (t_wmap_l_.z() + kSubCubeSideLengthHalf < 0)
    subcube_index_k--;

  // 如果对应的 subcube 处于整个 cube 的边缘, 则进行处理
  while (subcube_index_i < 3) {
    for (int j = 0; j < kCubeHeightNum; j++) {
      for (int k = 0; k < kCubeDepthNum; k++) {
        int i = kCubeWidthNum - 1;
        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_ptr =
          subcube_corner_array_[getArrayIndex(i, j, k)];
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_ptr = subcube_surf_array_[getArrayIndex(i, j, k)];
        for (; i >= 1; i--) {
          subcube_corner_array_[getArrayIndex(i, j, k)] =
            subcube_corner_array_[getArrayIndex(i - 1, j, k)];
          subcube_surf_array_[getArrayIndex(i, j, k)] =
            subcube_surf_array_[getArrayIndex(i - 1, j, k)];
        }
        subcube_corner_array_[getArrayIndex(i, j, k)] = corner_ptr;
        subcube_surf_array_[getArrayIndex(i, j, k)] = surf_ptr;
        corner_ptr->clear();
        surf_ptr->clear();
      }
    }

    subcube_index_i++;
    zero_in_cube_width_++;
  }

  while (subcube_index_i >= kCubeWidthNum - 3) {
    for (int j = 0; j < kCubeHeightNum; j++) {
      for (int k = 0; k < kCubeDepthNum; k++) {
        int i = 0;
        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_ptr =
          subcube_corner_array_[getArrayIndex(i, j, k)];
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_ptr = subcube_surf_array_[getArrayIndex(i, j, k)];
        for (; i < kCubeWidthNum - 1; i++) {
          subcube_corner_array_[getArrayIndex(i, j, k)] =
            subcube_corner_array_[getArrayIndex(i + 1, j, k)];
          subcube_surf_array_[getArrayIndex(i, j, k)] =
            subcube_surf_array_[getArrayIndex(i + 1, j, k)];
        }
        subcube_corner_array_[getArrayIndex(i, j, k)] = corner_ptr;
        subcube_surf_array_[getArrayIndex(i, j, k)] = surf_ptr;
        corner_ptr->clear();
        surf_ptr->clear();
      }
    }

    subcube_index_i--;
    zero_in_cube_width_--;
  }

  while (subcube_index_j < 3) {
    for (int i = 0; i < kCubeWidthNum; i++) {
      for (int k = 0; k < kCubeDepthNum; k++) {
        int j = kCubeHeightNum - 1;
        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_ptr =
          subcube_corner_array_[getArrayIndex(i, j, k)];
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_ptr = subcube_surf_array_[getArrayIndex(i, j, k)];
        for (; j >= 1; j--) {
          subcube_corner_array_[getArrayIndex(i, j, k)] =
            subcube_corner_array_[getArrayIndex(i, j - 1, k)];
          subcube_surf_array_[getArrayIndex(i, j, k)] =
            subcube_surf_array_[getArrayIndex(i, j - 1, k)];
        }
        subcube_corner_array_[getArrayIndex(i, j, k)] = corner_ptr;
        subcube_surf_array_[getArrayIndex(i, j, k)] = surf_ptr;
        corner_ptr->clear();
        surf_ptr->clear();
      }
    }

    subcube_index_j++;
    zero_in_cube_height_++;
  }

  while (subcube_index_j >= kCubeHeightNum - 3) {
    for (int i = 0; i < kCubeWidthNum; i++) {
      for (int k = 0; k < kCubeDepthNum; k++) {
        int j = 0;
        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_ptr =
          subcube_corner_array_[getArrayIndex(i, j, k)];
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_ptr = subcube_surf_array_[getArrayIndex(i, j, k)];
        for (; j < kCubeHeightNum - 1; j++) {
          subcube_corner_array_[getArrayIndex(i, j, k)] =
            subcube_corner_array_[getArrayIndex(i, j + 1, k)];
          subcube_surf_array_[getArrayIndex(i, j, k)] =
            subcube_surf_array_[getArrayIndex(i, j + 1, k)];
        }
        subcube_corner_array_[getArrayIndex(i, j, k)] = corner_ptr;
        subcube_surf_array_[getArrayIndex(i, j, k)] = surf_ptr;
        corner_ptr->clear();
        surf_ptr->clear();
      }
    }

    subcube_index_j--;
    zero_in_cube_height_--;
  }

  while (subcube_index_k < 3) {
    for (int i = 0; i < kCubeWidthNum; i++) {
      for (int j = 0; j < kCubeHeightNum; j++) {
        int k = kCubeDepthNum - 1;
        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_ptr =
          subcube_corner_array_[getArrayIndex(i, j, k)];
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_ptr = subcube_surf_array_[getArrayIndex(i, j, k)];
        for (; k >= 1; k--) {
          subcube_corner_array_[getArrayIndex(i, j, k)] =
            subcube_corner_array_[getArrayIndex(i, j, k - 1)];
          subcube_surf_array_[getArrayIndex(i, j, k)] =
            subcube_surf_array_[getArrayIndex(i, j, k - 1)];
        }
        subcube_corner_array_[getArrayIndex(i, j, k)] = corner_ptr;
        subcube_surf_array_[getArrayIndex(i, j, k)] = surf_ptr;
        corner_ptr->clear();
        surf_ptr->clear();
      }
    }

    subcube_index_k++;
    zero_in_cube_depth_++;
  }

  while (subcube_index_k >= kCubeDepthNum - 3) {
    for (int i = 0; i < kCubeWidthNum; i++) {
      for (int j = 0; j < kCubeHeightNum; j++) {
        int k = 0;
        pcl::PointCloud<pcl::PointXYZI>::Ptr corner_ptr =
          subcube_corner_array_[getArrayIndex(i, j, k)];
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf_ptr = subcube_surf_array_[getArrayIndex(i, j, k)];
        for (; k < kCubeDepthNum - 1; k++) {
          subcube_corner_array_[getArrayIndex(i, j, k)] =
            subcube_corner_array_[getArrayIndex(i, j, k + 1)];
          subcube_surf_array_[getArrayIndex(i, j, k)] =
            subcube_surf_array_[getArrayIndex(i, j, k + 1)];
        }
        subcube_corner_array_[getArrayIndex(i, j, k)] = corner_ptr;
        subcube_surf_array_[getArrayIndex(i, j, k)] = surf_ptr;
        corner_ptr->clear();
        surf_ptr->clear();
      }
    }

    subcube_index_k--;
    zero_in_cube_depth_--;
  }

  // 构建地图, 以当前位置所在的 subcube 进行获取, 左右各 2 个, 前后各 2 个, 上下各 1 个
  corner_map_.clear();
  surf_map_.clear();
  for (int i = subcube_index_i - 2; i <= subcube_index_i + 2; i++) {
    for (int j = subcube_index_j - 2; j <= subcube_index_j + 2; j++) {
      for (int k = subcube_index_k - 1; k <= subcube_index_k + 1; k++) {
        if (i >= 0 && i < kCubeWidthNum && j >= 0 && j < kCubeHeightNum && k >= 0 &&
            k < kCubeDepthNum) {
          corner_map_ += *subcube_corner_array_[getArrayIndex(i, j, k)];
          surf_map_ += *subcube_surf_array_[getArrayIndex(i, j, k)];
        }
      }
    }
  }
  return;
}

void LaserMapping::optimizePoseAndPushCube(
  const pcl::PointCloud<pcl::PointXYZI> &corner_point_less_sharp,
  const pcl::PointCloud<pcl::PointXYZI> &surf_point_less_flat)
{
  // 下采样边缘点及平面点
  pcl::PointCloud<pcl::PointXYZI> corner_point_stack;
  pcl::PointCloud<pcl::PointXYZI> surf_point_stack;
  downsize_filter_corner_.setInputCloud(corner_point_less_sharp.makeShared());
  downsize_filter_corner_.filter(corner_point_stack);
  downsize_filter_surf_.setInputCloud(surf_point_less_flat.makeShared());
  downsize_filter_surf_.filter(surf_point_stack);

  int corner_point_stack_size = corner_point_stack.size();
  int surf_point_stack_size = surf_point_stack.size();

  // 构建优化问题并优化位姿
  if (corner_map_.size() > 10 && surf_map_.size() > 50) {
    kdtree_corner_map_->setInputCloud(corner_map_.makeShared());
    kdtree_surf_map_->setInputCloud(surf_map_.makeShared());

    // kdtree 搜索使用变量
    pcl::PointXYZI point_after_transform;
    std::vector<int> point_search_index;
    std::vector<float> point_search_square_distance;

    // 迭代优化处理
    for (size_t iter_counter = 0; iter_counter < kMaxNumIterations; iter_counter++) {
      // ceres 问题构造
      ceres::Problem::Options problem_options;
      problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
      problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
      ceres::Problem problem(problem_options);
      problem.AddParameterBlock(q_wmap_l_.coeffs().data(), 4, eigen_quaternion_manifold_.get());
      problem.AddParameterBlock(t_wmap_l_.data(), 3);

      // 构造边缘约束项
      for (int i = 0; i < corner_point_stack_size; ++i) {
        // 为了方便搜索, 假设当前帧和上一帧的相对位姿相等, 将点转到上一帧的坐标系下
        point_after_transform = corner_point_stack[i];
        Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3) =
          (q_wmap_l_ *
             Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3).cast<double>() +
           t_wmap_l_)
            .cast<float>();

        // kdtree 搜索
        kdtree_corner_map_->nearestKSearch(point_after_transform, 5, point_search_index,
                                           point_search_square_distance);

        // 要求最远的点的距离小于 1m
        if (point_search_square_distance.back() < 1.f) {

          // 计算 5 个点的平均值并存储在 vector 中
          std::vector<Eigen::Vector3d> corners(5);
          Eigen::Vector3d corners_mean(0, 0, 0);
          for (int j = 0; j < 5; ++j) {
            corners[j] = Eigen::Map<const Eigen::Vector4f>(corner_map_[point_search_index[j]].data)
                           .head(3)
                           .cast<double>();
            corners_mean += corners[j];
          }
          corners_mean /= 5;

          // 计算 5 个点的协方差矩阵
          Eigen::Matrix3d corners_covariance_matrix;
          corners_covariance_matrix.setZero();
          for (int j = 0; j < 5; ++j) {
            corners_covariance_matrix +=
              (corners[j] - corners_mean) * (corners[j] - corners_mean).transpose();
          }

          // 求解特征值及特征向量, 如果满足最大特征值大于三倍的第二大特征值, 则构造 costfunction
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(corners_covariance_matrix);
          if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
            // 计算直线的主方向
            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
            Eigen::Vector3d curr_point =
              Eigen::Map<Eigen::Vector4f>(corner_point_stack[i].data).head(3).cast<double>();
            // 给定直线的两个点
            Eigen::Vector3d point_a = 0.1 * unit_direction + corners_mean;
            Eigen::Vector3d point_b = -0.1 * unit_direction + corners_mean;

            ceres::CostFunction *cost_function =
              LidarEdgeFactor::Create(curr_point, point_a, point_b);
            problem.AddResidualBlock(cost_function, huber_loss_function_.get(),
                                     q_wmap_l_.coeffs().data(), t_wmap_l_.data());
          }
        }
      }

      // 处理平面约束项
      for (int i = 0; i < surf_point_stack_size; ++i) {

        // 为了方便搜索, 假设当前帧和上一帧的相对位姿相等, 将点转到上一帧的坐标系下
        point_after_transform = surf_point_stack[i];
        Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3) =
          (q_wmap_l_ *
             Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3).cast<double>() +
           t_wmap_l_)
            .cast<float>();

        // kdtree 搜索
        kdtree_surf_map_->nearestKSearch(point_after_transform, 5, point_search_index,
                                         point_search_square_distance);

        // 要求最远的点的距离小于 1m
        if (point_search_square_distance.back() < 1.f) {

          // 根据搜索到的点拟合平面
          Eigen::Matrix<double, 5, 3> mat_a0;
          Eigen::Matrix<double, 5, 1> mat_b0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

          for (int j = 0; j < 5; ++j) {
            mat_a0.row(j) = Eigen::Map<const Eigen::Vector4f>(surf_map_[point_search_index[j]].data)
                              .head(3)
                              .cast<double>()
                              .transpose();
          }
          Eigen::Vector3d normal_vector = mat_a0.colPivHouseholderQr().solve(mat_b0);
          double negative_OA_dot_norm = 1 / normal_vector.norm();
          normal_vector.normalize();

          // 将点投回平面判定拟合好坏
          bool plane_valid = true;
          for (int j = 0; j < 5; j++) {
            if (fabs(normal_vector(0) * surf_map_[point_search_index[j]].x +
                     normal_vector(1) * surf_map_[point_search_index[j]].y +
                     normal_vector(2) * surf_map_[point_search_index[j]].z + negative_OA_dot_norm) >
                0.2) {
              plane_valid = false;
              break;
            }
          }

          // 构造 costfunction, 加入优化
          if (plane_valid) {
            Eigen::Vector3d curr_point =
              Eigen::Map<Eigen::Vector4f>(surf_point_stack[i].data).head(3).cast<double>();
            ceres::CostFunction *cost_function =
              LidarPlaneNormFactor::Create(curr_point, normal_vector, negative_OA_dot_norm);
            problem.AddResidualBlock(cost_function, huber_loss_function_.get(),
                                     q_wmap_l_.coeffs().data(), t_wmap_l_.data());
          }
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
  } else {
    std::cout << "局部地图特征点数量不足, 无法进行优化" << std::endl;
  }

  // 存储点云到 cube 中
  std::set<int> subcube_index_for_filter;
  for (int i = 0; i < corner_point_stack_size; i++) {
    // 为了方便搜索, 假设当前帧和上一帧的相对位姿相等, 将点转到上一帧的坐标系下
    pcl::PointXYZI point_after_transform = corner_point_stack[i];
    Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3) =
      (q_wmap_l_ * Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3).cast<double>() +
       t_wmap_l_)
        .cast<float>();

    // 计算属于哪个 subcube
    int subcube_index_i =
      int((point_after_transform.x + kSubCubeSideLengthHalf) / kSubCubeSideLength) +
      zero_in_cube_width_;
    int subcube_index_j =
      int((point_after_transform.y + kSubCubeSideLengthHalf) / kSubCubeSideLength) +
      zero_in_cube_height_;
    int subcube_index_k =
      int((point_after_transform.z + kSubCubeSideLengthHalf) / kSubCubeSideLength) +
      zero_in_cube_depth_;

    if (point_after_transform.x + kSubCubeSideLengthHalf < 0)
      subcube_index_i--;
    if (point_after_transform.y + kSubCubeSideLengthHalf < 0)
      subcube_index_j--;
    if (point_after_transform.z + kSubCubeSideLengthHalf < 0)
      subcube_index_k--;

    // 满足条件则进行存储
    if (subcube_index_i >= 0 && subcube_index_i < kCubeWidthNum && subcube_index_j >= 0 &&
        subcube_index_j < kCubeHeightNum && subcube_index_k >= 0 &&
        subcube_index_k < kCubeDepthNum) {
      int subcube_index = getArrayIndex(subcube_index_i, subcube_index_j, subcube_index_k);
      subcube_corner_array_[subcube_index]->push_back(point_after_transform);
      subcube_index_for_filter.insert(subcube_index);
    }
  }

  for (int i = 0; i < surf_point_stack_size; i++) {
    // 为了方便搜索, 假设当前帧和上一帧的相对位姿相等, 将点转到上一帧的坐标系下
    pcl::PointXYZI point_after_transform = surf_point_stack[i];
    Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3) =
      (q_wmap_l_ * Eigen::Map<Eigen::Vector4f>(point_after_transform.data).head(3).cast<double>() +
       t_wmap_l_)
        .cast<float>();

    // 计算属于哪个 subcube
    int subcube_index_i =
      int((point_after_transform.x + kSubCubeSideLengthHalf) / kSubCubeSideLength) +
      zero_in_cube_width_;
    int subcube_index_j =
      int((point_after_transform.y + kSubCubeSideLengthHalf) / kSubCubeSideLength) +
      zero_in_cube_height_;
    int subcube_index_k =
      int((point_after_transform.z + kSubCubeSideLengthHalf) / kSubCubeSideLength) +
      zero_in_cube_depth_;

    if (point_after_transform.x + kSubCubeSideLengthHalf < 0)
      subcube_index_i--;
    if (point_after_transform.y + kSubCubeSideLengthHalf < 0)
      subcube_index_j--;
    if (point_after_transform.z + kSubCubeSideLengthHalf < 0)
      subcube_index_k--;

    // 满足条件则进行存储
    if (subcube_index_i >= 0 && subcube_index_i < kCubeWidthNum && subcube_index_j >= 0 &&
        subcube_index_j < kCubeHeightNum && subcube_index_k >= 0 &&
        subcube_index_k < kCubeDepthNum) {
      int subcube_index = getArrayIndex(subcube_index_i, subcube_index_j, subcube_index_k);
      subcube_surf_array_[subcube_index]->push_back(point_after_transform);
      subcube_index_for_filter.insert(subcube_index);
    }
  }

  // 对 subcube 进行下采样
  for (auto &subcube_index : subcube_index_for_filter) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr subcube_corner(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr subcube_surf(new pcl::PointCloud<pcl::PointXYZI>());

    downsize_filter_corner_.setInputCloud(subcube_corner_array_[subcube_index]);
    downsize_filter_corner_.filter(*subcube_corner);
    subcube_corner_array_[subcube_index] = subcube_corner;

    downsize_filter_surf_.setInputCloud(subcube_surf_array_[subcube_index]);
    downsize_filter_surf_.filter(*subcube_surf);
    subcube_surf_array_[subcube_index] = subcube_surf;
  }
}

void LaserMapping::showLaserMapping()
{
  if (visualizer_ptr_ == nullptr) {
    return;
  }
  static int frame_count = 0;

  Eigen::Affine3f affine = Eigen::Affine3f::Identity();
  affine.translate(t_wmap_l_.cast<float>());
  affine.rotate(q_wmap_l_.toRotationMatrix().cast<float>());
  visualizer_ptr_->addCoordinateSystem(1, affine);

  // 显示过多会导致内存占用过多, 所以这里进行了精简处理, 减少内存占用
  if (frame_count % 4 == 0) {

    // 当前激光雷达位置处于哪个 subcube 中, 获取其坐标
    int subcube_index_i =
      int((t_wmap_l_.x() + kSubCubeSideLengthHalf) / kSubCubeSideLength) + zero_in_cube_width_;
    int subcube_index_j =
      int((t_wmap_l_.y() + kSubCubeSideLengthHalf) / kSubCubeSideLength) + zero_in_cube_height_;
    int subcube_index_k =
      int((t_wmap_l_.z() + kSubCubeSideLengthHalf) / kSubCubeSideLength) + zero_in_cube_depth_;

    if (t_wmap_l_.x() + kSubCubeSideLengthHalf < 0)
      subcube_index_i--;
    if (t_wmap_l_.y() + kSubCubeSideLengthHalf < 0)
      subcube_index_j--;
    if (t_wmap_l_.z() + kSubCubeSideLengthHalf < 0)
      subcube_index_k--;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cube_point_cloud_ptr(
      new pcl::PointCloud<pcl::PointXYZI>());
    for (int i = subcube_index_i - 2; i < subcube_index_i + 2; i++) {
      for (int j = subcube_index_j - 2; j < subcube_index_j + 2; j++) {
        for (int k = subcube_index_k - 2; k < subcube_index_k + 2; k++) {
          if (i >= 0 && i < kCubeWidthNum && j >= 0 && j < kCubeHeightNum && k >= 0 &&
              k < kCubeDepthNum) {
            *cube_point_cloud_ptr += *subcube_corner_array_[getArrayIndex(i, j, k)];
            *cube_point_cloud_ptr += *subcube_surf_array_[getArrayIndex(i, j, k)];
          }
        }
      }
    }

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> gray(cube_point_cloud_ptr, 192,
                                                                          192,
                                                                          192); // rgb
    visualizer_ptr_->addPointCloud<pcl::PointXYZI>(cube_point_cloud_ptr, gray,
                                                   "laser mapping" + std::to_string(frame_count));
  }

  ++frame_count;
}

} // namespace loam