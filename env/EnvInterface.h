#pragma once
#include <string>
#include <torch/torch.h>

namespace libforce{
namespace env{
class EnvInterface {
    public:

        virtual std::tuple<torch::Tensor, int64_t, bool> reset() = 0;
        virtual std::tuple<torch::Tensor, int64_t, bool> step(torch::Tensor) = 0;
        virtual torch::Tensor render() = 0;
        virtual void close() = 0;
};
}
}