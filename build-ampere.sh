#!/bin/bash
# Build for NVIDIA Ampere GPUs (A100, RTX 3090, RTX 4090, RTX 6000)
# Compute Capabilities: 80, 86, 87

export CMAKE_CUDA_ARCHITECTURES="80;86;87"
bash build.sh
