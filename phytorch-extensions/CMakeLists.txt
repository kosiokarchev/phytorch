cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)

project(phytorch LANGUAGES C CXX CUDA)


find_package(Python3 COMPONENTS Development.Module)
include_directories(${Python3_INCLUDE_DIRS})

set(Torch_DIR ~/.local/lib/python3.9/site-packages/torch/share/cmake/Torch)

set(CUDA_ROOT /opt/cuda)
find_package(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

set(CUDNN_LIBRARY_PATH /opt/cudnn/lib/libcudnn.so)
set(CUDNN_INCLUDE_PATH /opt/cudnn/include)
find_package(Torch)

message("${TORCH_INCLUDE_DIRS}")
message("${TORCH_LIBRARIES}")
include_directories(${TORCH_INCLUDE_DIRS})
link_libraries(${TORCH_LIBRARIES})


add_library(cephes SHARED cephes/cgamma.c cephes/hyp2f1.c)
add_library(common INTERFACE common/Loops.h common/complex.h common/implement.cuh common/utils.h common/implement.h common/templating.h common/preprocessor.h common/moremath.h)
add_library(ellipr SHARED ellipr/ellipr.cpp ellipr/ellipr_impl.cu ellipr/ellipr.cuh ellipr/elliprc.cu ellipr/elliprd.cu ellipr/elliprf.cu ellipr/elliprg.cu ellipr/elliprj.cu)
add_library(roots SHARED roots/roots.cpp roots/roots_impl.cu)
add_library(special SHARED special/gamma.cu special/hyper.cu special/special.cuh special/special.cpp special/special_impl.cu special/zeta.cu special/loggamma.cu special/digamma.cu special/polygamma.cu special/gammahead.h)


set_target_properties(ellipr PROPERTIES PREFIX "")
