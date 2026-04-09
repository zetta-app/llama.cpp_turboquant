git clone https://github.com/zetta-app/llama.cpp.git
git clone https://github.com/zetta-app/TurboQuant

rm -rf llama.cpp/build
cmake -B llama.cpp/build -S llama.cpp -DGGML_CUDA=ON -DLLAMA_TURBOQUANT=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build llama.cpp/build -j$(nproc)
