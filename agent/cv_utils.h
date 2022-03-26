#pragma once

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <torch/torch.h>
namespace libforce
{
  namespace utils
  {

    cv::Mat tensor2mat(torch::Tensor tensor){
        cv::Mat mat(tensor.size(0),tensor.size(1), CV_8UC3, tensor.to(torch::kCPU).data<uint8_t>());
        return mat;
    }
  
    cv::Mat resize_mat(cv::Mat mat, double fx, double fy){
        cv::resize(mat, mat, cv::Size(), fx, fy, cv::InterpolationFlags::INTER_LINEAR);
        return mat;
    }

    torch::Tensor mat2tensor(cv::Mat mat){
      return torch::from_blob(mat.data, {mat.rows, mat.cols, 3}, torch::kUInt8);
    }

    torch::Tensor HWCtoNCHW(torch::Tensor tensor){
      return tensor.unsqueeze(0).permute({0, 3, 1, 2});
    }

      /*
    cv::Mat tensor2Mat(torch::Tensor &tensor) {
	int height = tensor.size(0), width = tensor.size(1);
	//i_tensor = i_tensor.to(torch::kF32);
	tensor = tensor.to(torch::kCPU);
	cv::Mat mat(cv::Size(tensor.size(0), tensor.size(1)), CV_8UC4, tensor.to(torch::kCPU).data_ptr());
	return mat;
}*/
    /*
   cv::cvtColor(frame, frame, CV_BGR2RGB);
 //normalization
 frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
 //opencv format H*W*C
 auto input_tensor = torch::from_blob(frame.data, {1, frame_h, frame_w, kCHANNELS});
 //pytorch format N*C*H*W
 input_tensor = input_tensor.permute({0, 3, 1, 2});
  */

  }
}