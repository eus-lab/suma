#pragma once
#include <pybind11/numpy.h>
#include <torch/torch.h>

#include "agent/Spaces.h"

namespace libforce::utils
{

    std::vector<Info> space2vec(pybind11::object space)
    {
        auto gym_spaces = pybind11::module::import("gym.spaces");

        if (pybind11::isinstance(space, gym_spaces.attr("Box")))
        {
            auto shape = space.attr("shape").cast<int>();
            auto type = utils::get_dtype(space.attr("dtype").str().cast<std::string>());
            std::vector<Info> vec;
            for (auto i = 0; i < shape; i++)
            {
                Info info;
                info.sizes = std::vector{1};
                info.type = type;
                float low[1] = {space.attr("low").cast<float>()};
                info.low = torch::from_blob(low, {1,}, torch::kFloat32);
                float high[1] = {space.attr("high").cast<float>()};
                info.high = torch::from_blob(high, {1,}, torch::kFloat32);

                vec.push_back(info);
            }
            return vec;
        }
        else if (pybind11::isinstance(space, gym_spaces.attr("Discrete")))
        {
            auto n = space.attr("n").cast<int>();

            Info info;
            info.sizes = std::vector{1};
            info.type = utils::get_dtype(space.attr("dtype").str().cast<std::string>());
            auto low = 0;
            auto high = n - 1;

            return std::vector<Info>{info};
        }
        else if (pybind11::isinstance(space, gym_spaces.attr("Tuple")))
        {
            auto tuple = pybind11::tuple(space);
            auto n = pybind11::len(tuple);
            std::vector<Info> vec;
            for (auto i = 0; i < n; i++)
            {
                auto svec = space2vec(tuple[n]);
                vec.insert(vec.end(), svec.begin(), svec.end());
            }
            return vec;
        }
        else if (pybind11::isinstance(space, gym_spaces.attr("Dict")))
        {
            auto dict = pybind11::dict(space.attr("spaces"));
            std::vector<Info> vec;

            auto avec = space2vec(dict["achieved_goal"]);    // 3
            vec.insert(vec.end(), avec.begin(), avec.end());

            auto dvec = space2vec(dict["desired_goal"]);     // 3
            vec.insert(vec.end(), dvec.begin(), dvec.end());

            auto ovec = space2vec(dict["observation"]);      // ...
            vec.insert(vec.end(), ovec.begin(), ovec.end());

            return vec;
        }
        else if (pybind11::isinstance(space, gym_spaces.attr("MultiDiscrete")))
        {
            assert( false );
        }
        else if (pybind11::isinstance(space, gym_spaces.attr("MultiBinary")))
        {
            assert( false );
        }
        else{
            assert( false );
        }
    }

}