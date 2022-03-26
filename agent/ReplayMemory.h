#pragma once
#include <torch/torch.h>
#include <experimental/algorithm>
#include "cv_utils.h"

namespace libforce::modules

{
  struct ImageTransition
  {
    ImageTransition(){};
    ImageTransition(const ImageTransition & obj){
      screen0 = obj.screen0.clone();
      screen1 = obj.screen1.clone();
      screen2 = obj.screen2.clone();
      action = obj.action.clone();
      reward = obj.reward.clone();
      done = obj.done.clone();
    };

    std::vector<torch::Tensor> get_tensors()
    {
      return std::vector{
          screen0,
          screen1,
          screen2,
          action,
          reward,
          done};
    }
    void set_tensors(std::vector<torch::Tensor> tensors)
    {
      screen0 = tensors[0];
      screen1 = tensors[1];
      screen2 = tensors[2];
      action = tensors[3];
      reward = tensors[4];
      done = tensors[5];
    }
    

    void set_screen(torch::Tensor tensor)
    {
      screen2 = screen1;
      screen1 = screen0;
      screen0 = tensor.unsqueeze(0);
    }
    void set_action(torch::Tensor tensor)
    {
      action = tensor.unsqueeze(0);
    }
    void set_reward(torch::Tensor tensor)
    {
      reward = tensor.unsqueeze(0);
    }
    void set_done(torch::Tensor tensor)
    {
      done = tensor.unsqueeze(0);
    }

    torch::Tensor get_current_state()
    {
      return ((batch_resize(screen1) - batch_resize(screen0)) / 255);
    }
    torch::Tensor get_next_state()
    {
      return ((batch_resize(screen2) - batch_resize(screen1)) / 255);
    }
    torch::Tensor get_action()
    {
      return action;
    }
    torch::Tensor get_reward()
    {
      return reward;
    }
    torch::Tensor get_done()
    {
      return done;
    }

    static ImageTransition stack(std::vector<ImageTransition> inputs)
    {
      const size_t size = 6;
      std::array<std::vector<torch::Tensor>, size> results;
      for (auto input : inputs)
      {
        auto tensors = input.get_tensors();
        for (int i = 0; i < size; i++)
        {
          results[i].push_back(tensors[i].squeeze(0));
        }
      }
      std::vector<torch::Tensor> stacked;
      for (auto result : results)
      {
        stacked.push_back(torch::stack(result));
      }
      auto transition = ImageTransition();
      transition.set_tensors(stacked);
      return transition;
    }

  private:
    torch::Tensor screen0 = torch::zeros(1);
    torch::Tensor screen1 = torch::zeros(1);
    torch::Tensor screen2 = torch::zeros(1);
    torch::Tensor action = torch::zeros(1);
    torch::Tensor reward = torch::zeros(1);
    torch::Tensor done = torch::zeros(1);
    
    torch::Tensor batch_resize(torch::Tensor input)
    {
      std::vector<torch::Tensor> vec;
      for (auto i = 0; i < input.size(0); i++){
        vec.push_back(tensor_resize(input[i]));
      }
      return torch::stack(vec).clone();
    }
    torch::Tensor tensor_resize(torch::Tensor tensor)
    {
      // input: CHW tensor
      auto permuted = tensor.permute({1,2,0}); // CHW to HWC
      auto mat = utils::tensor2mat(permuted);
      auto resized = utils::resize_mat(mat, 0.1, 0.1);
      auto output = utils::mat2tensor(resized);
      return output.permute({2,0,1});        // HWC to CHW
    }
  };

  template <typename Transition>
  class ReplayMemory
  {
  public:
    ReplayMemory(){};
    ReplayMemory(size_t capacity_) : capacity(capacity_) {}

    void append(Transition transition)
    {
      memory.push_back(transition);
      std::cout << "m:"<<size() << std::endl;
      if (memory.size() > capacity)
      {
        memory.pop_front();
        memory.shrink_to_fit();
      }
    }

    Transition get_batch(size_t batch_size)
    {
      auto dataset = MemoryDataset(memory)
                         .map(torch::data::transforms::BatchLambda<std::vector<Transition>, Transition>(Transition::stack));

      auto data_loader = torch::data::make_data_loader(std::move(dataset), batch_size);
      Transition batch_transition;
      for (auto data : *data_loader)
      {
        batch_transition = data;
        break;
      }
      return batch_transition;
    }

    size_t size()
    {
      return memory.size();
    }

  private:
    std::deque<Transition> memory;
    size_t capacity;
    class MemoryDataset : public torch::data::Dataset<MemoryDataset, Transition>
    {
    public:
      // DQNMemoryDataset(){};
      MemoryDataset(std::deque<Transition>& memory_) : memory(memory_){};

      Transition get(size_t index) override
      {
        return memory[index];
      }

      torch::optional<size_t> size() const override
      {
        return memory.size();
      }

    private:
      std::deque<Transition> memory;
    };
  };

}