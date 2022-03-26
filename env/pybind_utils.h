#pragma once
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include <variant>

namespace libforce
{
    namespace utils
    {
        //----------------------------------------------------------
        template <typename T>
        static torch::ScalarType get_dtype();

        template <typename numpyT>
        torch::Tensor _pyobj2tensor(pybind11::object pyobj);

        template <typename numpyT>
        pybind11::array_t<numpyT> _tensor2numpy(torch::Tensor tensor);

        //----------------------------------------------------------
        // implementation

        template <typename T>
        static torch::ScalarType get_dtype()
        {
            if (std::is_same<T, uint8_t>())
            {
                return torch::kUInt8;
            }
            else if (std::is_same<T, int8_t>())
            {
                return torch::kInt8;
            }
            else if (std::is_same<T, int16_t>())
            {
                return torch::kInt16;
            }
            else if (std::is_same<T, int32_t>())
            {
                return torch::kInt32;
            }
            else if (std::is_same<T, int64_t>())
            {
                return torch::kInt64;
                //} else if (std::is_same<T, half>()){
                //    return torch::kFloat16;
            }
            else if (std::is_same<T, float>())
            {
                return torch::kFloat32;
            }
            else if (std::is_same<T, double>())
            {
                return torch::kFloat64;
            }
        }

        
        static torch::ScalarType get_dtype(std::string type)
        {
            if (type == "uint8")
            {
                return torch::kUInt8;
            }
            else if (type == "int8")
            {
                return torch::kInt8;
            }
            else if (type == "int16")
            {
                return torch::kInt16;
            }
            else if (type == "int32")
            {
                return torch::kInt32;
            }
            else if (type == "int64")
            {
                return torch::kInt64;
            }
            else if (type == "float32")
            {
                return torch::kFloat32;
            }
            else if (type == "float64")
            {
                return torch::kFloat64;
            }
        }

        // usage  <decltype(utils::to<kInt32>::type)> -> int32_t
        template <auto Val>
        struct to
        {
            static constexpr bool type{};
        };

        template <>
        struct to<torch::kUInt8>
        {
            static constexpr uint8_t type{};
        };
        template <>
        struct to<torch::kInt8>
        {
            static constexpr int8_t type{};
        };
        template <>
        struct to<torch::kInt16>
        {
            static constexpr int16_t type{};
        };
        template <>
        struct to<torch::kInt32>
        {
            static constexpr int32_t type{};
        };
        template <>
        struct to<torch::kInt64>
        {
            static constexpr int64_t type{};
        };
        template <>
        struct to<torch::kFloat32>
        {
            static constexpr float type{};
        };
        template <>
        struct to<torch::kFloat64>
        {
            static constexpr double type{};
        };

        std::variant<uint8_t, int8_t, int16_t, int32_t, int64_t, float, double> cast(torch::ScalarType type){
            if (type == torch::kUInt8)
            {
                return static_cast<uint8_t>(0);
            }
            else if (type == torch::kInt8)
            {
                return static_cast<int8_t>(0);
            }
            else if (type == torch::kInt16)
            {
                return static_cast<int16_t>(0);
            }
            else if (type == torch::kInt32)
            {
                return static_cast<int32_t>(0);
            }
            else if (type == torch::kInt64)
            {
                return static_cast<int64_t>(0);
            }
            else if (type == torch::kFloat32)
            {
                return static_cast<float>(0);
            }
            else if (type == torch::kFloat64)
            {
                return static_cast<double>(0);
            }
        }

        torch::Tensor pyobj2tensor(std::string type, pybind11::object pyobj)
        {
            if (type == "uint8")
            {
                return _pyobj2tensor<uint8_t>(pyobj);
            }
            else if (type == "int8")
            {
                return _pyobj2tensor<int8_t>(pyobj);
            }
            else if (type == "int16")
            {
                return _pyobj2tensor<int16_t>(pyobj);
            }
            else if (type == "int32")
            {
                return _pyobj2tensor<int32_t>(pyobj);
            }
            else if (type == "int64")
            {
                return _pyobj2tensor<int64_t>(pyobj);
            }
            else if (type == "float32")
            {
                return _pyobj2tensor<float>(pyobj);
            }
            else if (type == "float64")
            {
                return _pyobj2tensor<double>(pyobj);
            }
        }

        pybind11::array tensor2numpy(std::string type, torch::Tensor tensor)
        {
            if (type == "uint8")
            {
                return _tensor2numpy<uint8_t>(tensor).cast<pybind11::array>();
            }
            else if (type == "int8")
            {
                return _tensor2numpy<int8_t>(tensor).cast<pybind11::array>();
            }
            else if (type == "int16")
            {
                return _tensor2numpy<int16_t>(tensor).cast<pybind11::array>();
            }
            else if (type == "int32")
            {
                return _tensor2numpy<int32_t>(tensor).cast<pybind11::array>();
            }
            else if (type == "int64")
            {
                return _tensor2numpy<int64_t>(tensor).cast<pybind11::array>();
            }
            else if (type == "float32")
            {
                return _tensor2numpy<float>(tensor).cast<pybind11::array>();
            }
            else if (type == "float64")
            {
                return _tensor2numpy<double>(tensor).cast<pybind11::array>();
            }
        }
        
        template <typename numpyT>
        torch::Tensor _pyobj2tensor(pybind11::object pyobj)
        {

            torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

            auto numpy = pyobj.cast<pybind11::array_t<numpyT>>();
            std::vector<int64_t> shape;
            for (int i = 0; i < numpy.ndim(); i++)
            {
                shape.push_back(numpy.shape(i));
            }
            return torch::from_blob((void *)numpy.data(), shape, torch::TensorOptions().dtype(get_dtype<numpyT>()).device(device)).clone();
        }

        template <typename numpyT>
        pybind11::array_t<numpyT> _tensor2numpy(torch::Tensor tensor)
        {

            std::vector<numpyT> strides;
            for (int i = 0; i < tensor.dim(); i++)
            {
                strides.push_back(tensor.stride(i) * sizeof(numpyT));
            }
            return pybind11::array_t<numpyT>(tensor.sizes(), strides, (numpyT *)tensor.data_ptr(), pybind11::handle());
        }

    } // utils
} // libforce