cmake_minimum_required(VERSION 3.0.0)
project(LibForce VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)

include(CTest)
enable_testing()

list(APPEND CMAKE_PREFIX_PATH "../libtorch")
find_package(Torch REQUIRED)

add_library(libforce INTERFACE)

target_include_directories(
    libforce
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


#add_executable(testmain testmain.cpp)



add_subdirectory(env)

target_link_libraries(libforce INTERFACE env ${TORCH_LIBRARIES})


#target_link_libraries(testmain force)


add_library(libforce::agent ALIAS agent)
add_library(libforce::env ALIAS env)


set(DOC OFF)
if(DOC)
find_package(Doxygen REQUIRED dot)
set(DOXYGEN_GENERATE_LATEX ON)
doxygen_add_docs(doc_gen ./
            ALL
            USE_STAMP_FIFE)
endif()
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
