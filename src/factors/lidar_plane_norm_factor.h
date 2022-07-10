#ifndef LOAM_FACTORS_LIDAR_PLANE_NORM_FACTOR_H_
#define LOAM_FACTORS_LIDAR_PLANE_NORM_FACTOR_H_

#include "Eigen/Eigen"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace loam {

struct LidarPlaneNormFactor {
  LidarPlaneNormFactor(const Eigen::Vector3d &curr_point, const Eigen::Vector3d &plane_unit_norm,
                       double negative_OA_dot_norm)
    : curr_point_(curr_point), plane_unit_norm_(plane_unit_norm),
      negative_OA_dot_norm_(negative_OA_dot_norm)
  {
  }

  template <typename T> bool operator()(const T *q, const T *t, T *residual) const
  {

    auto cp = curr_point_.cast<T>();
    auto norm = plane_unit_norm_.cast<T>();

    Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};

    Eigen::Matrix<T, 3, 1> point_w = q_w_curr * cp + t_w_curr;

    residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm_);

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point,
                                     const Eigen::Vector3d &plane_unit_norm,
                                     const double negative_OA_dot_norm)
  {
    return (new ceres::AutoDiffCostFunction<LidarPlaneNormFactor, 1, 4, 3>(
      new LidarPlaneNormFactor(curr_point, plane_unit_norm, negative_OA_dot_norm)));
  }

  Eigen::Vector3d curr_point_;
  Eigen::Vector3d plane_unit_norm_;
  double negative_OA_dot_norm_;
};
} // namespace loam

#endif