#pragma once
#include <torch/torch.h>
#include "AgentInterface.h"
#include "DQNModel.h"
#include "ReplayMemory.h"
#include "DQNActor.h"

namespace libforce::agent
{

  template <class Transition = modules::ImageTransition, class Memory = modules::ReplayMemory<Transition>, class Model = modules::DQNmodel>
  class DQN : AgentInterface
  {
  public:
    DQN();
    void reset(torch::Tensor) override;
    torch::Tensor action(torch::Tensor) override;
    void update(torch::Tensor, torch::Tensor) override;

  private:
    void push_memory(torch::Tensor, torch::Tensor);

    Model policy_net = nullptr;
    Model target_net = nullptr;

    Transition transition;
    //std::shared_ptr<Memory> memory;
    //std::shared_ptr<decltype(Memory().map(modules::Transition::funcs::stack))> memory=nullptr;
    Memory memory;

    DQNActor<Transition, Model, torch::optim::RMSprop> actor;

    std::unique_ptr<torch::optim::RMSprop> optimizer;

    torch::Tensor current_state;
    torch::Tensor next_state;
    torch::Tensor current_action;

    //modules::ScreenSubtract screen_subtract;

    size_t n_actions;
    size_t steps_done;

    int BATCH_SIZE = 128;
    float GAMMA = 0.999;
    float EPS_START = 0.9;
    float EPS_END = 0.05;
    int EPS_DECAY = 200;
    int TARGET_UPDATE = 10;
  };

  template <class Transition, class Memory, class Model>
  DQN<Transition, Memory, Model>::DQN()
  //:memory(std::move(std::make_shared<Memory>(10000)->map(modules::Transition::funcs::stack)))
  {
    //policy_net = Model(40, 60, 2);
    //target_net = Model(40, 60, 2);

    //torch::save(policy_net, "policy_net_state.pt");
    //torch::load(target_net, "policy_net_state.pt");

    //target_net->eval();
    actor = DQNActor<Transition, Model, torch::optim::RMSprop>();

    transition = Transition();

    memory = Memory(500);
    //memory = std::move(std::make_shared<Memory>(10000)->map(modules::Transition::funcs::stack));
    //auto mory = Memory(10000).map(modules::Transition::funcs::stack);
    //memory.map(modules::Transition::funcs::stack);
    //memory->reset(std::move(Memory(10000)->map(modules::Transition::funcs::stack)));
    n_actions = 2;

    //optimizer = std::move(std::make_unique<torch::optim::RMSprop>(policy_net->parameters()));

    steps_done = 0;
  }
  template <class Transition, class Memory, class Model>
  void DQN<Transition, Memory, Model>::reset(torch::Tensor screen)
  {
    transition.set_screen(screen);
    transition.set_screen(screen);
    steps_done = 0;
  }

  template <class Transition, class Memory, class Model>
  torch::Tensor DQN<Transition, Memory, Model>::action(torch::Tensor screen)
  {
    // input: state
    // return: action

    transition.set_screen(screen);
    auto current_state = transition.get_current_state();

    //global steps_done;
    auto action = actor.action(current_state);
    
    transition.set_action(action);
    return action.clone();
  }
  
  template <class Transition, class Memory, class Model>
  void DQN<Transition, Memory, Model>::update(torch::Tensor reward, torch::Tensor done)
  {
    
    transition.set_reward(reward);
    transition.set_done(done);
    memory.append(transition);

    auto batch_transition = memory.get_batch(BATCH_SIZE);
    auto size = memory.size();
    actor.update(size, batch_transition);
  }
} // libforce::agent