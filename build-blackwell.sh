#!/bin/bash
# Build for NVIDIA Blackwell GPUs (H200, L40S, RTX 6000 Ada)
# Compute Capability: 12.0

export CMAKE_CUDA_ARCHITECTURES=120
bash build.sh
