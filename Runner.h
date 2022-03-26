#pragma once
#include "Agent.h"
#include "Env.h"

#include <memory>
#include <ostream>

namespace libforce
{
    template <class Agent, class Env>
    requires std::is_base_of_v<agent::AgentInterface, Agent> && std::is_base_of_v<env::EnvInterface, Env>
    class Runner
    {
    public:
        Runner();
        Runner(Agent agent_, Env env_);
        void run(size_t episodes);


    private:
        std::unique_ptr<Env> env;
        std::unique_ptr<Agent> agent;
    };

    template <class Agent, class Env>
    requires std::is_base_of_v<agent::AgentInterface, Agent> && std::is_base_of_v<env::EnvInterface, Env>
    Runner<Agent, Env>::Runner()
    {
        env = std::move(std::make_unique<Env>());
        agent = std::move(std::make_unique<Agent>());
    }

    template <class Agent, class Env>
    Runner<Agent, Env>::Runner(Agent agent_, Env env_)
    {
        //env = std::move(env_);
        //agent = std::move(agent_);
    }

    template <class Agent, class Env>
    void Runner<Agent, Env>::run(size_t episodes)
    {
        for (size_t i_episode = 0; i_episode < episodes; i_episode++)
        {
            std::cout << i_episode << std::endl;
        
            // Initialize the environment and state
            env->reset();
            auto screen = env->render();
            agent->reset(screen);

            size_t count = 0;
            while (true)
            {
                auto screen = env->render();
                
                auto action = agent->action(screen);
                
                auto result = env->step(action);

                agent->update(torch::tensor({std::get<1>(result)}), torch::tensor({std::get<2>(result)}));

                if (std::get<2>(result))
                {
                    break;
                }

            }
        }
        //env->reset();
        //env->close();
    }

} // libforce