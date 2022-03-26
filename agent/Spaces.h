#pragma once

#include <torch/torch.h>

namespace libforce
{
  struct Info{
      std::vector<int> sizes;
      torch::ScalarType type;
      torch::Tensor high;
      torch::Tensor low;
  };
  struct Spaces{
      std::vector<Info> action;
      std::vector<Info> observation;
  };
  struct MujocoSpaces : Spaces{
      std::vector<Info> achieved_goal;
      std::vector<Info> desired_goal;
  };


}