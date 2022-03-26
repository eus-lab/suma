#pragma once
#include "EnvInterface.h"
#include "gym_envs.h"
#include <string>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <torch/torch.h>

#include "pybind_utils.h"
#include <tuple>
#include <cassert>
#include <memory>
#include <ostream>

#include "agent/Spaces.h"
#include "spaces_utils.h"


//auto destroyer = [](pybind11::object o){o.release();};
namespace libforce{
namespace env{

template<gym envid>
class Gym : EnvInterface {
    public:
        Gym();
        ~Gym();
        //~Environment() = default; 
        Gym(Gym&&) = default; 
        //Environment& operator=(Environment&&) = default;
        //virtual ~Environment() = default;

        std::tuple<torch::Tensor, int64_t, bool> reset() override;
        std::tuple<torch::Tensor, int64_t, bool> step(torch::Tensor) override;
        torch::Tensor render() override;
        void close() override;

        Spaces get_info();
    private:
        std::unique_ptr<pybind11::object/*, decltype(destroyer)*/> env;
        
        //std::unique_ptr<pybind11::scoped_interpreter> python;
        
        torch::ScalarType observation_space_dtype;
        torch::ScalarType action_space_dtype = torch::kInt16;

        std::string observation_space_str;
        std::string action_space_str;

        std::tuple<torch::Tensor, int64_t, bool> unpack_observation_list(pybind11::object);      

        Spaces spaces;

        bool is_rgb;
};



template<gym envid>
Gym<envid>::Gym()/* : python(std::move(std::make_unique<pybind11::scoped_interpreter>()))*/{
    //pybind11::gil_scoped_acquire guard;
    pybind11::initialize_interpreter();
    auto env_name = gymids[static_cast<int>(envid)];
    auto gym = pybind11::module::import("gym");

    is_rgb = true;

    // make env
    env = std::move(std::make_unique<pybind11::object>(gym.attr("make")(env_name)));

    /*
    Spaces spaces;
    spaces.action = utils::space2vec(env->attr("action_space"));
    if (is_rgb){
        Info info;
        auto image = render();
        info.sizes = image.sizes();
        info.type = image.dtype();
        info.low = 0;
        info.high = 255;
        spaces.observation.push_back(info);
    } else {
        spaces.observation = utils::space2vec(env->attr("observation_space"));
    }
    this.spaces = spaces;
    */

    // set torch::ScalarType
    action_space_str = env->attr("action_space").attr("dtype").str().cast<std::string>();
    action_space_dtype = utils::get_dtype(action_space_str);

    observation_space_str = env->attr("observation_space").attr("dtype").str().cast<std::string>();
    observation_space_dtype = utils::get_dtype(observation_space_str);
}

template<gym envid>
Gym<envid>::~Gym(){
    //auto r = env->release();
    //pybind11::finalize_interpreter();
    //pybind11::gil_scoped_release guard;
    //auto p = std::move(python);
    //orch::isinf
    //pybind11::gil_scoped_release();
}


template<gym envid>
Spaces Gym<envid>::get_info(){
    return spaces;
}

template<gym envid>
std::tuple<torch::Tensor, int64_t, bool> Gym<envid>::reset(){
    auto state = env->attr("reset")();
    return unpack_observation_list(state);
}

// 
template<gym envid>
std::tuple<torch::Tensor, int64_t, bool> Gym<envid>::step(torch::Tensor action){
    auto state = pybind11::object();
    
    auto action_array = utils::tensor2numpy(action_space_str, action);
    //auto actary = std::visit([&](auto x){ utils::_tensor2numpy<decltype(x)>(action);}, utils::cast(action_space_dtype));
    if(action_array.size() == 1){
        // tensor to item 
        state = env->attr("step")(action_array.index_at(0));
    } else{
        state = env->attr("step")(action_array);
    }
    return unpack_observation_list(state);
}

template<gym envid>
torch::Tensor Gym<envid>::render(){

    //auto screen = env->attr("render")("rgb_array").attr("transpose")(2,0,1);
    //auto type = env->attr("render")("rgb_array").attr("dtype").str().cast<std::string>();
    //return utils::pyobj2tensor(type, screen).clone();

    //auto screen = env->attr("render")("rgb_array").attr("transpose")(2,0,1).attr("astype")("float32");
    //auto screen = env->attr("render")("rgb_array");
    
    auto type = env->attr("render")("rgb_array").attr("dtype").str().cast<std::string>();
    //auto type = "uint8";
    auto screen = env->attr("render")("rgb_array").attr("astype")(type);
    //auto type = screen.attr("dtype").str().cast<std::string>();

    return utils::pyobj2tensor(type, screen).permute({2,0,1}).clone();
}

template<gym envid>
void Gym<envid>::close(){
    env->attr("close")();
}

template<gym envid>
std::tuple<torch::Tensor, int64_t, bool> Gym<envid>::unpack_observation_list(pybind11::object obj){
    auto pylist = obj.cast<pybind11::list>();

    auto aryobj = pylist[0].attr("astype")(observation_space_str);
    auto state = utils::pyobj2tensor(observation_space_str, aryobj);
    //auto state = std::visit([&](auto x){ utils::_pyobj2tensor<decltype(x)>(obj);}, utils::cast(observation_space_dtype));
    
    
    auto reward = pylist[1].cast<double>();
    
    auto finish = pylist[2].cast<bool>();

    return std::make_tuple(state.clone(), reward, finish);
}



} // env
} // libforce