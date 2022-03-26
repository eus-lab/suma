#add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
cmake_minimum_required(VERSION 3.0.0)
project(env VERSION 0.1.0)

include(CTest)
enable_testing()

add_library(env INTERFACE)
add_library(force::env ALIAS env)

target_include_directories(
    env
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


add_subdirectory(../agent agent)


find_package (Python3 COMPONENTS Development NumPy)
target_link_libraries(${PROJECT_NAME} INTERFACE Python3::Python Python3::NumPy agent)


find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} INTERFACE ${TORCH_LIBRARIES})

#install(TARGETS env EXPORT env-export)
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include/env)



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
