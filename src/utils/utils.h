#ifndef LOAM_UTILS_UTILS_H_
#define LOAM_UTILS_UTILS_H_

#include <cmath>
namespace utils {
namespace angle {

/*!
 * @brief 输入角度, 转为弧度
 * @param degree 输入角度
 * @return 转换之后的弧度
 */
inline double degree2Rad(double degree) { return degree * M_PI / 180.0; }

/*!
 * @brief 输入弧度, 转为角度
 * @param rad 输入弧度
 * @return 转换之后的角度
 */
inline double rad2Degree(double rad) { return rad * 180.0 / M_PI; }

/*!
 * @brief 将角度转到 0~2π 之间
 * @param angle 输入角度, 要求为弧度
 * @return 转换之后的角度
 */
template <typename T> inline T convertAngleTo2Pi(T angle)
{
  while (angle > 2 * M_PI)
    angle -= 2 * M_PI;
  while (angle < 0)
    angle += 2 * M_PI;
  return angle;
}

/*!
 * @brief 将角度转到 -π~π 之间
 * @param angle 输入角度, 要求为弧度
 * @return 转换之后的角度
 */
template <typename T> inline T convertAngleToPi(T angle)
{
  if (angle > M_PI)
    angle -= 2 * M_PI;
  else if (angle < -M_PI)
    angle += 2 * M_PI;
  return angle;
}

} // namespace angle

namespace pcl {

/*!
 * @brief 计算点云数据到原点距离的平方
 * @param pt 输入的点云数据
 * @return 距离的平方
 */
template <typename PT> inline double DistanceSquare(const PT &pt)
{
  return pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
}

/*!
 * @brief 计算两个点云数据之间距离的平方
 * @param pt1 第一个点云数据
 * @param pt2 第二个点云数据
 * @return 距离的平方
 */
template <typename PT> inline double DistanceSquare(const PT &pt1, const PT &pt2)
{
  return (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y) +
         (pt1.z - pt2.z) * (pt1.z - pt2.z);
}

/*!
 * @brief 计算点云数据到原点距离
 * @param pt 输入的点云数据
 * @return 距离
 */
template <typename PT> inline double Distance(const PT &pt) { return sqrt(DistanceSquare(pt)); }

/*!
 * @brief 计算两个点云数据之间距离
 * @param pt1 第一个点云数据
 * @param pt2 第二个点云数据
 * @return 距离
 */
template <typename PT> inline double Distance(const PT &pt1, const PT &pt2)
{
  return sqrt(DistanceSquare(pt1, pt2));
}

} // namespace pcl
} // namespace utils

#endif