#pragma once
#include <torch/torch.h>
#include <experimental/algorithm>
#include "cv_utils.h"

namespace libforce::modules

{
  /*
struct Transitions
{
  torch::Tensor screen_0;
  torch::Tensor screen_1;
  torch::Tensor screen_2;
  torch::Tensor reward;
  torch::Tensor done;
};*/
/*
  struct DQNTransition
  {
    DQNTransition(){};
    enum key
    {
      screen0,
      screen1,
      screen2,
      action,
      reward,
      done
    };
    std::map<key, torch::Tensor> dict;

    torch::Tensor get_current_state()
    {
      return dict[key::screen0] - dict[key::screen1];
    }
    torch::Tensor get_next_state()
    {
      return dict[key::screen1] - dict[key::screen2];
    }
    torch::Tensor get_action()
    {
      return dict[key::action];
    }
    torch::Tensor get_reward()
    {
      return dict[key::reward];
    }
    torch::Tensor get_done()
    {
      return dict[key::done];
    }

    void resize()
    {
      dict[key::screen0] = funcs::tensor_resize(dict[key::screen0]);
      dict[key::screen1] = funcs::tensor_resize(dict[key::screen1]);
      dict[key::screen2] = funcs::tensor_resize(dict[key::screen2]);
    }

    struct funcs
    {
      static DQNTransition stack(std::vector<DQNTransition> inputs)
      {
        std::map<key, std::vector<torch::Tensor>> output;
        for (auto input : inputs)
        {
          for (auto &[key, value] : input.dict)
          {
            output[key].push_back(tensor_resize(value));
          }
        }
        DQNTransition transition{};
        for (auto [key, value] : output)
        {
          transition.dict[key] = torch::stack(value);
        }
        return transition;
      }
      static torch::Tensor tensor_resize(torch::Tensor tensor)
      {
        auto mat = utils::tensor2mat(tensor);
        auto resized = utils::resize_mat(mat, 0.1, 0.1);
        auto output = utils::mat2tensor(resized);
        return utils::HWCtoNCHW(output);
      }
    };
  };
  */

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

  /*
 class DQNReplayMemory{
    public:
    DQNReplayMemory(){};
    DQNReplayMemory(size_t capacity_) : capacity(capacity_) {}

    void append(DQNTransition transition){
      memory.push_back(transition);
    if (memory.size() > capacity)
    {
      memory.pop_front();
      memory.shrink_to_fit();
    }
    }

    DQNTransition get_batch(size_t batch_size){
      //auto dataset = DQNMemoryDataset(memory).map(DQNTransition::funcs::stack);
      auto dataset = DQNMemoryDataset(memory);
      auto data_loader = torch::data::make_data_loader(dataset, batch_size);
      DQNTransition batch_transition;
      for (auto data : *data_loader){
        batch_transition = data[0];
        break;
      }
      return batch_transition;
    }
    private:
    std::deque<DQNTransition> memory;
    size_t capacity;
 };
*/
  /*
    class DQNMemoryDataset : public torch::data::Dataset<DQNMemoryDataset, DQNTransition>
    {
    public:
      DQNMemoryDataset(){};
      DQNMemoryDataset(std::deque<DQNTransition> memory_) : memory(memory_){};

      DQNTransition get(size_t index) override
      {
        return memory[index];
      }

      torch::optional<size_t> size() const override
      {
        return memory.size();
      }

    private:
      std::deque<DQNTransition> memory;
    };
    */
  /*
template<typename Transition>
 class MemoryDataset : public torch::data::Dataset<MemoryDataset<Transition>, Transition>{
   public:
   MemoryDataset(std::deque<Transition> memory_) : memory(memory_){};

    Transition get(size_t index) override {
      return memory[index];
    }

    torch::optional<size_t> size() const override {
      return memory.size();
    }
    private:
    std::deque<Transition> memory;
 };
*/

}