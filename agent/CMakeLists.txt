cmake_minimum_required(VERSION 3.0.0)
project(agent VERSION 0.1.0)

include(CTest)
enable_testing()

add_library(agent INTERFACE)
add_library(force::agent ALIAS agent)


target_include_directories(
    agent
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

find_package(Torch REQUIRED)
target_link_libraries(agent INTERFACE ${TORCH_LIBRARIES})


# For OpenCV
find_package(OpenCV COMPONENTS imgproc) #4.5.1
target_include_directories(agent INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(agent INTERFACE ${OpenCV_LIBS})


#install(TARGETS agent EXPORT agent-export)
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include/agent)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
