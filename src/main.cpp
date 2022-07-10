
#include "laser_mapping.h"
#include "laser_odometry.h"
#include "pcl/io/pcd_io.h"
#include "pcl/visualization/cloud_viewer.h"
#include "scan_registration.h"
#include "utils/command_line_parser.h"
#include "utils/filesystem.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

struct PointCloudInfo {
  long timestamp;
  std::string path;
};

std::vector<PointCloudInfo> readPointCloudInfo(const std::string &point_cloud_dir)
{
  // 读取时间戳, 转为 utc 时间
  std::string timestamp_path = point_cloud_dir + "/timestamps.txt";
  std::ifstream ifs(timestamp_path, std::ifstream::in);
  std::vector<long> timestamps;
  std::string line;
  int year, month, day, hour, minute;
  double second;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    sscanf(line.c_str(), "%d-%d-%d %d:%d:%lf", &year, &month, &day, &hour, &minute, &second);
  }

  // 读取路径
  std::string data_dir = point_cloud_dir + "/data";
  std::vector<PointCloudInfo> point_cloud_infos;
  PointCloudInfo tmp;
  for (auto &it : utils::filesystem::directory_iterator(data_dir)) {
    if (utils::filesystem::is_regular_file(it.path()) &&
        utils::filesystem::detail::endsWith(it.path().string(), std::string(".bin"))) {
      tmp.path = it.path().string();
      point_cloud_infos.push_back(tmp);
    }
  }
  std::sort(point_cloud_infos.begin(), point_cloud_infos.end(),
            [](const PointCloudInfo &a, const PointCloudInfo &b) { return a.path < b.path; });
  return point_cloud_infos;
}

int main(int argc, char **argv)
{
  // 解析命令行
  utils::cmdline::Options options("loam");

  options.add_options()("dataset_dir", "dir that contain point cloud data",
                        utils::cmdline::value<std::string>());

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }
  std::string point_cloud_dir = result["dataset_dir"].as<std::string>();

  // 获取所有点云数据信息
  if (!utils::filesystem::exists(point_cloud_dir)) {
    exit(0);
  }
  std::vector<PointCloudInfo> point_cloud_infos = readPointCloudInfo(point_cloud_dir);

  // 创建显示并构建特征提取&激光雷达里程计&激光雷达建图
  pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer());
  visualizer->setWindowName("loam");
  visualizer->setCameraPosition(0, 0, 200, 0, 0, 0, 1, 0, 0, 0);
  visualizer->setSize(2500, 1500);
  loam::ScanRegistration scan_registration(visualizer);
  loam::LaserOdometry laser_odometry(visualizer);
  loam::LaserMapping laser_mapping(visualizer);

  // 加载点云
  bool thread_stop = false;
  std::thread thread([&]() {
    for (int i = 0; !thread_stop && i < point_cloud_infos.size(); ++i) {

      auto &info = point_cloud_infos[i];
      std::fstream input(info.path.c_str(), std::ios::in | std::ios::binary);
      if (!input.good()) {
        std::cerr << "Could not read file: " << info.path << std::endl;
        exit(EXIT_FAILURE);
      }
      input.seekg(0, std::ios::beg);

      pcl::PointCloud<pcl::PointXYZI>::Ptr points(new pcl::PointCloud<pcl::PointXYZI>);

      while (input.good() && !input.eof()) {
        pcl::PointXYZI point;
        input.read((char *)&point.x, 3 * sizeof(float));
        input.read((char *)&point.intensity, sizeof(float));
        points->push_back(point);
      }
      input.close();

      scan_registration.processPointClouds(*points);
      laser_odometry.processPointClouds(
        scan_registration.corner_points_sharp_, scan_registration.corner_points_less_sharp_,
        scan_registration.surf_points_flat_, scan_registration.surf_points_less_flat_, *points);

      if (i % 5 == 0) {
        laser_mapping.processPointClouds(scan_registration.corner_points_less_sharp_,
                                         scan_registration.surf_points_less_flat_, *points,
                                         laser_odometry.q_w_l_, laser_odometry.t_w_l_);
      }
    }
  });

  while (!visualizer->wasStopped()) {
    visualizer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  thread_stop = true;
  if (thread.joinable()) {
    thread.join();
  }

  return 0;
}