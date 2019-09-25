#设置cmake的最小版本
cmake_minimum_required(VERSION 3.5)
#项目名称
project(tf-demo)
#设置c++标准
set(CMAKE_CXX_STANDARD 11)
#设置tensorflow编译根目录
set(TENSORFLOW_DIR /root/projects/tf/tensorflow)

# opencv设置
include_directories(/root/projects/opencv3/include)
link_directories(/root/projects/opencv3/lib64)
# abseil
include_directories(/usr/local/abseil/include)
link_directories(/usr/local/abseil/lib/)
#项目中的include路径
include_directories(${TENSORFLOW_DIR})
include_directories(${TENSORFLOW_DIR}/bazel-genfiles)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/proto)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/proto)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/include)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/eigen)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/public)

#项目中lib路径
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/lib)
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/lib)
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11)
link_directories(${TENSORFLOW_DIR}/bazel-bin/tensorflow)

add_executable(tf_predictor main.cpp)
#连接libtensorflow_cc.so和libtensorflow_framework库
target_link_libraries(tf_predictor -ldl -lpthread tensorflow_cc tensorflow_framework opencv_videoio opencv_imgcodecs opencv_core opencv_imgproc opencv_highgui)
