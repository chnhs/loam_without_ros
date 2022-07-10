#ifndef LOAM_FACTORS_LIDAR_DISTANCE_FACTOR_H_
#define LOAM_FACTORS_LIDAR_DISTANCE_FACTOR_H_
#include "Eigen/Eigen"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace loam {
struct LidarDistanceFactor {

  LidarDistanceFactor(const Eigen::Vector3d &curr_point, const Eigen::Vector3d &closed_point)
    : curr_point_(curr_point), closed_point_(closed_point)
  {
  }

  template <typename T> bool operator()(const T *q, const T *t, T *residual) const
  {
    Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
    Eigen::Matrix<T, 3, 1> cp = curr_point_.cast<T>();
    Eigen::Matrix<T, 3, 1> point_w = q_w_curr * cp + t_w_curr;

    residual[0] = point_w.x() - T(closed_point_.x());
    residual[1] = point_w.y() - T(closed_point_.y());
    residual[2] = point_w.z() - T(closed_point_.z());
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point,
                                     const Eigen::Vector3d &closed_point)
  {
    return (new ceres::AutoDiffCostFunction<LidarDistanceFactor, 3, 4, 3>(
      new LidarDistanceFactor(curr_point, closed_point)));
  }

  Eigen::Vector3d curr_point_;
  Eigen::Vector3d closed_point_;
};
} // namespace loam

#endif