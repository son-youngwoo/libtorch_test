#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

int main(int argc, char** argv) {

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

  return 0;
}