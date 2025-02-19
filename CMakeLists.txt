project(FalezCSNNSimulator)
cmake_minimum_required(VERSION 3.1)

option(USE_GUI "Enable GUI" OFF)
# option(USE_GUI "Enable GUI" ON)

if(USE_GUI)
    message(STATUS "GUI Enable")
else()
    message(STATUS "GUI Disable")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(LIBRARY_NAME "CSNNS" CACHE STRING "Name of library")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(USE_GUI)
    set(CMAKE_AUTOMOC ON)
    find_package(Qt4 4.8.7 REQUIRED QtCore QtGui)
    add_definitions(-DENABLE_QT)

    include_directories(include/ ${QT_INCLUDES})
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

message(STATUS ${LAPACK_INCLUDES})

file(GLOB_RECURSE SOURCES src/* include/*)
list(FILTER SOURCES EXCLUDE REGEX "apps/benchmarks/Bench7\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "apps/benchmarks/BenchCifar10WP\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "apps/cifar/Cifar10_L1\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "apps/cifar/Cifar10_L2\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "apps/whitening/GenerateCifar10Whiten\\.cpp")
list(FILTER SOURCES EXCLUDE REGEX "apps/whitening/GenerateSTL10Whiten\\.cpp")
include_directories(include/ ${LAPACK_INCLUDES})
include_directories( ${OpenCV_INCLUDE_DIRS} )


set(APPS_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ansi -pedantic -Wshadow -Weffc++ -ftemplate-backtrace-limit=0 -ltbb2 -msse -msse2 -msse3 -march=native -mfpmath=sse")
# add_definitions(-DSMID_AVX256)

add_subdirectory(dep/libsvm)

if(USE_GUI)
    add_subdirectory(dep/qcustomplot)
endif()


#
#   Build library
#

add_library(${LIBRARY_NAME} SHARED ${SOURCES})
set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS ${APPS_FLAGS})
target_link_libraries(${LIBRARY_NAME} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} libsvm Threads::Threads ${OpenCV_LIBS} tbb)
if(USE_GUI)
   target_link_libraries(${LIBRARY_NAME} Qt4::QtCore Qt4::QtGui qcustomplot)
endif()

#   Build executables
#
file(GLOB_RECURSE apps apps/static_motion_modeling/*.cpp)
foreach(app ${apps})
    get_filename_component(app_name ${app} NAME_WE)
    add_executable(${app_name} ${app})
    set_target_properties(${app_name} PROPERTIES COMPILE_FLAGS ${APPS_FLAGS})
    target_link_libraries(${app_name} ${LIBRARY_NAME} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} Threads::Threads ${OpenCV_LIBS} tbb)
    #target_link_libraries(${app_name} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} Threads::Threads ${OpenCV_LIBS} tbb)
    if(USE_GUI)
	target_link_libraries(${app_name} ${LIBRARY_NAME} Qt4::QtCore Qt4::QtGui)
    endif()
endforeach()
