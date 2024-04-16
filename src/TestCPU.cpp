#include <ros/ros.h>
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>

int main(int argc, char** argv) {
  ros::init(argc, argv, "cpu_test_node");
  ros::NodeHandle nh("~");

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

    // CPU inference // 
    torch::Tensor input_tensor = torch::rand({518, 1, 100, 100}); // {batch_size, channel, height, width}

  while(ros::ok()) {
    auto start = std::chrono::high_resolution_clock::now();

    torch::Tensor output = module.forward({input_tensor}).toTensor();

    // std::cout << output << std::endl;
    ///////////////////

    // 시간을 측정할 코드 블록 끝
    auto end = std::chrono::high_resolution_clock::now();

    // 측정된 시간 계산
    std::chrono::duration<double> elapsed = end - start;

    // 결과 출력
    std::cout << "실행 시간: " << elapsed.count() << "초" << std::endl;

  }
  return 0;
}