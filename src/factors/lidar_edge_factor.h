#ifndef LOAM_FACTORS_LIDAR_EDGE_FACTOR_H_
#define LOAM_FACTORS_LIDAR_EDGE_FACTOR_H_

#include "Eigen/Eigen"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace loam {
struct LidarEdgeFactor {
  LidarEdgeFactor(const Eigen::Vector3d &curr_point, const Eigen::Vector3d &last_point_a,
                  const Eigen::Vector3d &last_point_b)
    : curr_point_(curr_point), last_point_a_(last_point_a), last_point_b_(last_point_b)
  {
  }

  template <typename T> bool operator()(const T *q, const T *t, T *residual) const
  {
    auto cp = curr_point_.cast<T>();
    auto lpa = last_point_a_.cast<T>();
    auto lpb = last_point_b_.cast<T>();

    Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_last_curr{t[0], t[1], t[2]};

    Eigen::Matrix<T, 3, 1> lp;
    lp = q_last_curr * cp + t_last_curr;

    Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
    Eigen::Matrix<T, 3, 1> de = lpa - lpb;

    residual[0] = nu.x() / de.norm();
    residual[1] = nu.y() / de.norm();
    residual[2] = nu.z() / de.norm();

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point,
                                     const Eigen::Vector3d &last_point_a,
                                     const Eigen::Vector3d &last_point_b)
  {
    return (new ceres::AutoDiffCostFunction<LidarEdgeFactor, 3, 4, 3>(
      new LidarEdgeFactor(curr_point, last_point_a, last_point_b)));
  }

  Eigen::Vector3d curr_point_, last_point_a_, last_point_b_;
};
} // namespace loam

#endif