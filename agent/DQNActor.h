#pragma once

#include <torch/torch.h>

namespace libforce::agent
{

    template <class Transition, class Model, class Optimizer>
    class DQNActor
    {
    public:
        DQNActor();

        Model policy_net = nullptr;
        Model target_net = nullptr;

        std::unique_ptr<Optimizer> optimizer;
        // torch::optim::RMSprop

        torch::Tensor action(torch::Tensor current_state);
        void update(size_t memory_size, Transition batch_transition);

        int BATCH_SIZE = 128;
        float GAMMA = 0.999;
        float EPS_START = 0.9;
        float EPS_END = 0.05;
        int EPS_DECAY = 200;
        int TARGET_UPDATE = 10;

        size_t n_actions;
        size_t steps_done;
    };

    template <class Transition, class Model, class Optimizer>
    DQNActor<Transition, Model, Optimizer>::DQNActor()
    {
        policy_net = Model(40, 60, 2);
        target_net = Model(40, 60, 2);

        torch::save(policy_net, "policy_net_state.pt");
        torch::load(target_net, "policy_net_state.pt");

        target_net->eval();

        optimizer = std::move(std::make_unique<Optimizer>(policy_net->parameters()));

        n_actions = 2;

        steps_done = 0;
    }

    template <class Transition, class Model, class Optimizer>
    torch::Tensor DQNActor<Transition, Model, Optimizer>::action(torch::Tensor current_state)
    {
        std::uniform_real_distribution<float> rand01(0.0, 1.0);
        auto mt = std::mt19937{std::random_device{}()};
        auto sample = rand01(mt);
        auto eps_threshold = EPS_END + (EPS_START - EPS_END) * std::exp(-1. * steps_done / EPS_DECAY);
        steps_done += 1;
        torch::Tensor action;
        if (sample > eps_threshold)
        {
            torch::NoGradGuard no_grad;
            action = std::get<1>(policy_net(current_state).max(1)).view({1});
        }
        else
        {
            std::uniform_int_distribution<> randrange(0, n_actions - 1);
            auto mt = std::mt19937{std::random_device{}()};
            uint64_t data[] = {static_cast<uint64_t>(randrange(mt))};
            action = torch::from_blob(data, {1}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
        }
        return action.clone();
    }

    template <class Transition, class Model, class Optimizer>
    void DQNActor<Transition, Model, Optimizer>::update(size_t memory_size, Transition batch_transition)
    {
        if (memory_size < BATCH_SIZE)
        {
            return;
        }

        // Compute a mask

        auto non_final_mask = batch_transition.get_done();

        auto next_state = batch_transition.get_next_state();
        std::vector<torch::Tensor> mask_vec;
        for (auto i = 0; i < BATCH_SIZE; i++)
        {
            if (non_final_mask[i].item().toBool())
            {
                mask_vec.push_back(next_state[i]);
            }
        }

        auto non_final_next_states = torch::stack(mask_vec); // 4
    
        auto state_batch = batch_transition.get_current_state();
        auto action_batch = batch_transition.get_action();
        auto reward_batch = batch_transition.get_reward();

        // Compute Q(s_t, a)
        auto state_action_values = policy_net(state_batch).gather(1, action_batch); // 2

        // Compute V(s_{t+1})
        auto next_state_values = torch::zeros({BATCH_SIZE, 1}, torch::TensorOptions().dtype(torch::kFloat)); // 2

        next_state_values.masked_scatter_(non_final_mask, std::get<0>(target_net(non_final_next_states).max(1)).detach()); // 2

        // Compute the expected Q values
        auto expected_state_action_values = (next_state_values * GAMMA) + reward_batch; //  200.0).clamp(-1., 1.); // 2

        // Compute Huber loss
        auto criterion = torch::nn::SmoothL1Loss();
        auto loss = criterion(state_action_values, expected_state_action_values);

        // Optimize the model
        optimizer->zero_grad();
        loss.backward();

        for (auto param : policy_net->parameters())
        {
            param.grad().data().clamp_(-1, 1);
        }

        optimizer->step();

        // Update the target network, copying all weights and biases in DQN
        if (steps_done % TARGET_UPDATE == 0)
        {
            torch::save(policy_net, "policy_net_state.pt");
            torch::load(target_net, "policy_net_state.pt");
        }
    }

}