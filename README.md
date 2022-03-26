# LibForce : C++ Library for Deep Reinforcement Learning


## Overview

LibForce is a Deep 
Reinforcement Learning library built using C++ and 
LibTorch (PyTorch C++ Frontend). 

## Getting Started




You will need:

* Ubuntu 20
* g++ >= 10.0.0
* CMake >= 3.0.0
* LibTorch = 10.0.0
  
If you use env component, 
you will need:
* Python
* gym[all]


Clone LibTorch like that:

    git clone URL

Download and extract LibTorch like that:

    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu.zip
    
    unzip libtorch-cxx11-abi-shared-with-deps-1.10.0+cpu.zip


Download [mjpro 150](https://roboti.us/download.html)
and
[activation key](https://roboti.us/license.html).
Extract like that:


    unzip mjpro150_xxx.zip
    mkdir ~/.mujoco
    mv mjpro150 ~/.mujoco
    mv /path/to/mjkey.txt ~/.mujoco/

Install Open AI Gym like that:

    pip install gym[all]


* [mujoco instalation troubleshooting](https://github.com/openai/mujoco-py#ubuntu-installtion-troubleshooting)
* sudo apt install swig patchelf

### CMake

Configure your CMakeLists.txt to link LibForce and LibTorch:

    list(APPEND CMAKE_PREFIX_PATH "/path/to/libtorch")
    add_subdirectory("/path/to/libforch" libforce)

    target_link_libraries(your_target_name libforce)


## Usage

    #include "libforce/LibForce.h"

    int main(void)
    {
      using namespace libforce;

      auto exec = Runner<agent::DQN<>, env::Gym<gym::CartPole_v0>>();
      exec.run(10000);
      return 0;
    }

## Implemented Algorithms

* DQN ([Reference](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html))