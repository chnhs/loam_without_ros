cmake_minimum_required(VERSION 3.12)

project(loam_without_ros)

# 产生带调试信息的编译后目标代码
add_definitions(-g)

# 显示全部 warring
add_definitions(-Wall)

# 探测到永不执行的代码时发出警告
add_definitions(-Wunreachable-code)

# 当局部变量遮盖另一局部变量或全局变量时发出警告
add_definitions(-Wshadow)

# 对函数指针或void *类型的指针进行算术操作时发出警告
add_definitions(-Wpointer-arith)

# 寻找第三方库(此处添加 Qt5_DIR 以及 VTK_DIR 是因为 brew install pcl 找不到这两个)
set(Qt5_DIR /usr/local/Cellar/qt@5/5.15.5/lib/cmake/Qt5)
set(VTK_DIR /usr/local/Cellar/vtk/9.1.0_4/lib/cmake/vtk-9.1)
find_package(PCL 1.12 REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)
find_package(Ceres 2.0.0 REQUIRED)

# 设置待编译文件
set(LOAM_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/scan_registration.cpp
    ${PROJECT_SOURCE_DIR}/src/laser_odometry.cpp
    ${PROJECT_SOURCE_DIR}/src/laser_mapping.cpp)

# 编译并链接三方库
add_definitions(${PCL_DEFINITIONS})
add_executable(loam ${LOAM_SOURCE_FILES})
target_link_libraries(loam ${PCL_LIBRARIES} Eigen3::Eigen Ceres::ceres)
target_include_directories(loam PRIVATE ${PCL_INCLUDE_DIRS})
