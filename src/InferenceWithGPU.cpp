// #include <ros/ros.h>
// #include <iostream>
// #include <torch/script.h>
// #include <torch/torch.h>
// #include <chrono>
// #include <opencv2/opencv.hpp>

// int main(int argc, char** argv) {
//   ros::init(argc, argv, "inference_gpu_node");
//   ros::NodeHandle nh("~");

//   torch::jit::script::Module module;

//   try {
//     // torch::jit::load()을 사용해 ScriptModule을 파일로부터 역직렬화
//     module = torch::jit::load("/home/orin/catkin_ws/src/libtorch_test/models/20231127(1)_libtorch.pt");
//   }
//   catch (const c10::Error& e) {
//     std::cerr << "error loading the model\n";
//     return -1;
//   }
//   std::cout << "ok\n";

//   cv::Mat patch(100, 100, CV_8U);
//   cv::randu(patch, cv::Scalar(0), cv::Scalar(256));
    
//   std::cout << "CV Data Type : " << patch.type() << std::endl;

//   torch::ScalarType dtype;
//   dtype = torch::kFloat32;

//   torch::Tensor patch_t = torch::from_blob(patch.data, {100, 100, patch.channels()}, dtype);
// //   std::cout << "patch_t size : " << patch_t.sizes() << std::endl;
  
//   patch_t = patch_t.unsqueeze(0); // {100, 100, 1} -> {1, 100, 100, 1}
//   patch_t = patch_t.permute({0, 3, 1, 2}); // {1, 100, 100, 1} -> {1, 1, 100,}
//   patch_t = patch_t.to(torch::kCUDA);
//   std::cout << "patch_t dtype : " << patch_t.dtype() << std::endl;
//   std::cout << "patch_t size : " << patch_t.sizes() << std::endl;

//   std::vector<torch::jit::IValue> patches;
//   patches.push_back(patch_t);

//   std::cout << "patches size : " << patches.size() << std::endl;

//   // GPU inference // 
//   // torch::Device device(torch::kCUDA);
  
//   torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

//   std::cout << device << std::endl;

//   module.to(torch::kCUDA);


//   torch::Tensor input_tensor = torch::rand({1, 1, 100, 100}).to(torch::kCUDA);  // {batch_size, channel, height, width}
//   std::cout << "input_tensor size : " << input_tensor.sizes() << std::endl;
//   std::cout << "input_tensor dtype : " << input_tensor.dtype() << std::endl;

//   while(ros::ok()) {
//     auto start = std::chrono::high_resolution_clock::now();
    
//     torch::Tensor output = module.forward({patch_t}).toTensor();

//     // std::cout << output << std::endl;
//     ///////////////////

//     // 시간을 측정할 코드 블록 끝
//     auto end = std::chrono::high_resolution_clock::now();

//     // 측정된 시간 계산
//     std::chrono::duration<double> elapsed = end - start;

//     // 결과 출력
//     std::cout << "실행 시간: " << elapsed.count() << "초" << std::endl;
//   }
//   return 0;
// }

//잘되는거
#include <ros/ros.h>
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <fstream>

int main(int argc, char** argv) {
  ros::init(argc, argv, "inference_gpu_node");
  ros::NodeHandle nh("~");

  std::ofstream outputFile1("/home/orin/catkin_ws/src/libtorch_test/data/patch.txt");
  std::ofstream outputFile2("/home/orin/catkin_ws/src/libtorch_test/data/patch_clone.txt");
  std::ofstream outputFile3("/home/orin/catkin_ws/src/libtorch_test/data/model_output.txt");

  torch::jit::script::Module module;

  try {
    // torch::jit::load()을 사용해 ScriptModule을 파일로부터 역직렬화
    module = torch::jit::load("/home/orin/catkin_ws/src/libtorch_test/models/20231127(1)_libtorch.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  std::cout << "ok\n";

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  module.to(torch::kCUDA);

  std::vector<torch::Tensor> patches;

  cv::Mat patch(100, 100, CV_32F);
  // cv::Mat patch(100, 100, CV_8U);
  torch::ScalarType dtype;
  dtype = torch::kFloat32;

  for (int i = 0; i < 2500; i++) {
    cv::randu(patch, cv::Scalar(255), cv::Scalar(255));
    // std::cout << patch.type() << std::endl;
    // outputFile1 << patch << std::endl;

    cv::Mat _patch = patch.clone();

    // _patch = _patch/255.0f;  
    // outputFile2 << _patch << std::endl;

    // std::cout << patch.type() << std::endl;

    // std::cout << patch << std::endl;
    // std::cout << patch << std::endl;
    torch::Tensor patch_t = torch::from_blob(_patch.data, {1, 100, 100}, dtype);
    patch_t = patch_t.div(255);
    std::cout << _patch.channels() << std::endl;
    // torch::Tensor patch_t = torch::from_blob(_patch.data, {100, 100, _patch.channels()}, dtype);
    // std::cout << patch_t << std::endl;
    // patch_t = patch_t.permute({2, 0, 1}); // {100, 100, 1} -> {1, 100, 100}
    // std::cout << patch_t.sizes() << std::endl;

    patches.push_back(patch_t);
  }
  // std::cout << patches.size() << std::endl;

  torch::Tensor batch = torch::stack(patches);
  batch = batch.to(torch::kCUDA);

  // std::cout << batch << std::endl;

  // while(ros::ok()) {
    // auto start = std::chrono::high_resolution_clock::now();
    
    torch::Tensor output = module.forward({batch}).toTensor();

    // std::cout << output << std::endl;
    outputFile3 << output << std::endl;
    // ///////////////////

    // // 시간을 측정할 코드 블록 끝
    // auto end = std::chrono::high_resolution_clock::now();

    // // 측정된 시간 계산
    // std::chrono::duration<double> elapsed = end - start;

    // // 결과 출력
    // std::cout << "실행 시간: " << elapsed.count() << "초" << std::endl;
  // }
  return 0;
}
