git clone https://github.com/zetta-app/llama.cpp
git clone https://github.com/zetta-app/quant.cpp

rm -rf llama.cpp/build

# Build with optional CUDA architecture specification
CMAKE_ARGS="-DGGML_CUDA=ON -DLLAMA_TURBOQUANT=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON"

if [ -n "$CMAKE_CUDA_ARCHITECTURES" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
fi

cmake -B llama.cpp/build -S llama.cpp $CMAKE_ARGS
cmake --build llama.cpp/build -j$(nproc)
