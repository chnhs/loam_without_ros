#ifndef LOAM_FACTORS_LIDAR_PLANE_FACTOR_H_
#define LOAM_FACTORS_LIDAR_PLANE_FACTOR_H_

#include "Eigen/Eigen"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace loam {

struct LidarPlaneFactor {
  LidarPlaneFactor(const Eigen::Vector3d &curr_point, const Eigen::Vector3d &last_point_j,
                   const Eigen::Vector3d &last_point_l, const Eigen::Vector3d &last_point_m)
    : curr_point_(curr_point), last_point_j_(last_point_j), last_point_l_(last_point_l),
      last_point_m_(last_point_m)
  {
    ljm_norm_ = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
    ljm_norm_.normalize();
  }

  template <typename T> bool operator()(const T *q, const T *t, T *residual) const
  {

    auto cp = curr_point_.cast<T>();
    auto lpj = last_point_j_.cast<T>();
    auto ljm = ljm_norm_.cast<T>();

    Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_last_curr{t[0], t[1], t[2]};

    Eigen::Matrix<T, 3, 1> lp;
    lp = q_last_curr * cp + t_last_curr;

    residual[0] = (lp - lpj).dot(ljm);

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point,
                                     const Eigen::Vector3d &last_point_j,
                                     const Eigen::Vector3d &last_point_l,
                                     const Eigen::Vector3d &last_point_m)
  {
    return (new ceres::AutoDiffCostFunction<LidarPlaneFactor, 1, 4, 3>(
      new LidarPlaneFactor(curr_point, last_point_j, last_point_l, last_point_m)));
  }

  Eigen::Vector3d curr_point_, last_point_j_, last_point_l_, last_point_m_;
  Eigen::Vector3d ljm_norm_;
};
} // namespace loam

#endif