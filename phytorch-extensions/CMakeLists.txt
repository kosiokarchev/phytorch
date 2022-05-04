cmake_minimum_required(VERSION 3.19)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)

project(phytorch LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

find_package(Python3 COMPONENTS Development.Module)
include_directories(${Python3_INCLUDE_DIRS})

find_package(Boost REQUIRED)
message("${Boost_INCLUDE_DIRS}")
message("${Boost_LIBRARIES}")
include_directories(${Boost_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES})

find_package(Torch)

message("${TORCH_INCLUDE_DIRS}")
message("${TORCH_LIBRARIES}")
include_directories(${TORCH_INCLUDE_DIRS})
link_libraries(${TORCH_LIBRARIES})


add_library(cephes SHARED cephes/cgamma.c cephes/hyp2f1.c)
add_library(
    common INTERFACE
    common/Loops.h
    common/complex.h
    common/implement.cuh
    common/utils.h
    common/implement.h
    common/templating.h
    common/preprocessor.h
    common/moremath.h
)
add_library(elliptic SHARED elliptic/elliptic.cpp elliptic/elliptic_impl.cu elliptic/elliptic.cuh elliptic/elliprc.cu elliptic/elliprd.cu elliptic/elliprf.cu elliptic/elliprg.cu elliptic/elliprj.cu elliptic/ellip.cu)
add_library(roots SHARED roots/roots.cpp roots/roots_impl.cu)
add_library(
    special SHARED
    special/special.cuh special/special.cpp special/special_impl.cu
        special/gammahead.h
        special/gamma.cu
        special/loggamma.cu
        special/digamma.cu
        special/polygamma.cu
        special/gammainc.cu
        special/gammaincinv.cu
    special/hyper.cu
    special/zeta.cu
)


set_target_properties(elliptic PROPERTIES PREFIX "")
