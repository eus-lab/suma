#pragma once
#include <torch/torch.h>

namespace libforce {
namespace agent{

class AgentInterface {
        public:
        virtual void reset(torch::Tensor)=0;
        virtual torch::Tensor action(torch::Tensor)=0;
        virtual void update(torch::Tensor, torch::Tensor)=0;
    
    //private:
        
};
} // agent
} // libforce