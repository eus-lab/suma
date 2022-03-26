#pragma once
#include <torch/torch.h>
#include <random>
namespace libforce::modules
{
struct DQNmodelImpl : torch::nn::Module
{
  DQNmodelImpl(int64_t h, int64_t w, int64_t outputs)
      : conv1(torch::nn::Conv2dOptions(3, 16, 5)
                  .stride(2)),
        batch_norm1(16),
        conv2(torch::nn::Conv2dOptions(16, 32, 5)
                  .stride(2)),
        batch_norm2(32),
        conv3(torch::nn::Conv2dOptions(32, 32, 5)
                  .stride(2)),
        batch_norm3(32)
  {
    // register_module() is needed if we want to use the parameters() method later on
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);

    // Number of Linear input connections depends on output of conv2d layers
    // and therefore the input image size, so compute it.
    auto convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)));
    auto convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)));
    auto linear_input_size = convw * convh * 32;
    head = torch::nn::Linear(linear_input_size, outputs);
    register_module("head", head);
  }
  int64_t conv2d_size_out(int64_t size, int64_t kernel_size = 5, int64_t stride = 2) //5,2
  {
    return (size - (kernel_size - 1) - 1) / stride + 1;
  }
  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(batch_norm1(conv1(x)));
    x = torch::relu(batch_norm2(conv2(x)));
    x = torch::relu(batch_norm3(conv3(x)));
    return head(x.view({x.size(0), -1}));
  }
  torch::nn::Linear head{nullptr};
  torch::nn::Conv2d conv1, conv2, conv3;
  torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(DQNmodel);


int BATCH_SIZE = 128;
float GAMMA = 0.999;
float EPS_START = 0.9;
float EPS_END = 0.05;
int EPS_DECAY = 200;
int TARGET_UPDATE = 10;

int steps_done = 0;
torch::Tensor select_action(torch::Tensor state, DQNmodel policy_net, size_t n_actions)
{
  //global steps_done;
  std::uniform_real_distribution<float> rand01(0.0, 1.0);
  auto mt = std::mt19937{std::random_device{}()};
  auto sample = rand01(mt);
  auto eps_threshold = EPS_END + (EPS_START - EPS_END) * std::exp(-1. * steps_done / EPS_DECAY);
  steps_done += 1;
  if (sample > eps_threshold)
  {
    torch::NoGradGuard no_grad;
    return std::get<1>(policy_net(state).max(1)).view({1, 1});
  }
  else
  {
    std::uniform_int_distribution<> randrange(0, n_actions - 1);
    auto mt = std::mt19937{std::random_device{}()};
    uint64_t data[] = {static_cast<uint64_t>(randrange(mt))};
    return torch::from_blob(data, {1, 1}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
  }
}
}