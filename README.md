# LLM Inference with KV Cache Compression

A workspace combining **llama.cpp** (production LLM inference engine) with **quant.cpp** (KV cache compression research platform). This setup enables running large language models locally with 4-7x memory compression while maintaining quality.

## Quick Start

### Build

```bash
bash build.sh
```

This clones both repositories and builds llama.cpp with TurboQuant KV compression support enabled.

### Run Server

```bash
bash run-server.sh
```

Launches an OpenAI-compatible API server on port 8080 with:
- **Model**: Qwen3-Coder-Next (9B parameters)
- **KV Compression**: `tq_uniform_4b` (7x compression)
- **Context**: 131K tokens
- **GPU**: 16 layers offloaded to NVIDIA GPU

### Test API

```bash
bash cli-test.sh
```

Sends a test chat completion request to the running server.

## Project Structure

```
.
├── llama.cpp/              # Production LLM inference engine
│   ├── tools/
│   │   ├── cli/            # llama-cli: interactive chat
│   │   ├── server/         # llama-server: OpenAI-compatible API
│   │   ├── perplexity/     # Quality measurement tool
│   │   └── llama-bench/    # Performance benchmarking
│   ├── ggml/               # Core tensor library
│   └── docs/               # Build guides, architecture docs
│
├── quant.cpp/              # KV cache compression research
│   ├── include/            # Public C API (quant.h)
│   ├── src/
│   │   ├── core/           # Quantization algorithms
│   │   ├── backend/        # CPU/GPU kernels
│   │   └── engine/         # GGUF loader, inference
│   ├── wasm/               # Browser demo (192KB)
│   ├── bench/              # Performance benchmarks
│   └── docs/               # API reference, guides
│
├── build.sh                # Build script (clone + cmake)
├── run-server.sh           # Start inference server
├── cli-test.sh             # Test API endpoint
└── README.md               # This file
```

## What Each Component Does

### llama.cpp

**Production-grade LLM inference engine** (250K+ LOC, C/C++)

- Supports 100+ model architectures (LLaMA, Qwen, Gemma, Mixtral, etc.)
- Multiple backends: Metal (Apple Silicon), CUDA (NVIDIA), HIP (AMD), Vulkan, CPU
- Tools:
  - `llama-cli`: Interactive chat with grammar constraints
  - `llama-server`: OpenAI-compatible REST API
  - `llama-bench`: Performance benchmarking
  - `llama-perplexity`: Quality measurement

**Key features used here:**
- CUDA acceleration (16 layers offloaded)
- Flash attention
- Jinja template support
- Speculative decoding ready

### quant.cpp

**Minimal KV cache compression engine** (72K LOC, pure C)

- 7 quantization schemes: `turbo_kv_*`, `uniform_*`, `polar_*`, `qjl_*`
- Compression: 4-7x memory reduction
- Quality: +0.7% to +5.7% PPL degradation (vs FP32 KV)
- Ships as single header (`quant.h`, 628KB) or full library
- Runs on: iOS, Android, WASM, MSVC, microcontrollers

**KV compression types:**
- `turbo_kv_5b`: 5.8x compression, +0.7% PPL (quality champion)
- `turbo_kv_4b`: 7.1x compression, +5.7% PPL (default, speed/quality balance)
- `uniform_4b`: 7.5x compression, +7.7% PPL (simple, no delta overhead)

## Configuration

### Server Settings (run-server.sh)

```bash
-hf unsloth/Qwen3-Coder-Next-GGUF:Qwen3-Coder-Next-UD-Q4_K_XL
  # Model: Qwen 3 Coder Next (9B), Q4_K_XL quantization

-ngl 16
  # GPU layers: 16 layers offloaded to NVIDIA GPU

-c 131072
  # Context window: 131K tokens

--cache-type-k tq_uniform_4b
  # KV compression: uniform 4-bit (7x compression)

-np 1
  # Parallel requests: 1 concurrent request

--flash-attn on
  # Flash attention: enabled for speed

--port 8080
  # API port: 8080
```

### Modify for Your Hardware

**For Apple Silicon (Metal):**
```bash
# In run-server.sh, replace:
# -ngl 16 \
# with:
# -ngl 999 \  # Offload all layers to Metal
```

**For AMD GPU (HIP):**
```bash
# Rebuild with HIP support:
cmake -B llama.cpp/build -S llama.cpp -DGGML_HIP=ON
```

**For CPU only:**
```bash
# In run-server.sh, remove:
# -ngl 16 \
```

## API Usage

### Chat Completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256
  }'
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-")

response = client.chat.completions.create(
    model="Qwen3.5-9B",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=256
)

print(response.choices[0].message.content)
```

### JavaScript/Node.js

```javascript
const response = await fetch("http://localhost:8080/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "Qwen3.5-9B",
    messages: [{ role: "user", content: "Hello" }],
    max_tokens: 256
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

## Performance

### Memory Usage (Llama 3.2 3B, 131K context)

| KV Format | Memory | Context | Compression |
|-----------|--------|---------|-------------|
| FP32 (baseline) | 4.2 GB | 50K tokens | 1× |
| FP16 (standard) | 2.1 GB | 100K tokens | 2× |
| `uniform_4b` | 600 MB | 350K tokens | **7× |
| `turbo_kv_4b` | 600 MB | 350K tokens | **7.1×** |

### Speed (Llama 3.2 3B, M1 Pro)

| Config | Throughput | vs FP32 KV |
|--------|-----------|-----------|
| FP32 KV (NEON) | 14.8 tok/s | baseline |
| `turbo_kv_4b` | 13.7 tok/s | −7.8% |
| `uniform_4b` | 11.7 tok/s | −21% |

## Building from Source

### Prerequisites

- CMake 3.15+
- C++17 compiler (GCC, Clang, MSVC)
- CUDA 11.8+ (for GPU support)
- Python 3.8+ (for model download)

### Build Steps

```bash
# 1. Clone and build
bash build.sh

# 2. Download a model (optional, run-server.sh does this automatically)
pip install huggingface_hub
hf download unsloth/Qwen3-Coder-Next-GGUF:Qwen3-Coder-Next-UD-Q4_K_XL --local-dir models/

# 3. Run server
bash run-server.sh

# 4. Test in another terminal
bash cli-test.sh
```

### Build Options

```bash
# Rebuild with different options
cmake -B llama.cpp/build -S llama.cpp \
  -DGGML_CUDA=ON \           # NVIDIA GPU
  -DGGML_METAL=ON \          # Apple Metal
  -DGGML_HIP=ON \            # AMD GPU
  -DLLAMA_TURBOQUANT=ON \    # KV compression
  -DCMAKE_BUILD_TYPE=Release

cmake --build llama.cpp/build -j$(nproc)
```

## Supported Models

Any GGUF-format model works. Popular choices:

| Model | Size | Speed (M1) | Context |
|-------|------|-----------|---------|
| SmolLM2 135M | 135M | 103 tok/s | 350K |
| Llama 3.2 3B | 3B | 10 tok/s | 350K |
| Qwen 3.5 4B | 4B | 20 tok/s | 350K |
| Llama 3.1 8B | 8B | 5 tok/s | 350K |
| Gemma 4 26B MoE | 26B | 3.9 tok/s | 350K |

Download from [Hugging Face](https://huggingface.co/models?library=gguf):

```bash
hf download <user>/<model>-GGUF --local-dir models/
```

## Documentation

- **[llama.cpp docs](llama.cpp/docs/)** — Build guides, architecture, API reference
- **[quant.cpp docs](quant.cpp/docs/)** — KV compression algorithms, custom quantization
- **[llama.cpp README](llama.cpp/README.md)** — Full feature list, bindings, UIs
- **[quant.cpp README](quant.cpp/README.md)** — Compression benchmarks, WASM demo

## Troubleshooting

### Out of Memory

Reduce context or use more aggressive compression:

```bash
# In run-server.sh:
-c 65536 \                    # Reduce context to 64K
--cache-type-k tq_uniform_3b  # Use 3-bit compression (9x)
```

### Slow Inference

Enable GPU acceleration:

```bash
# Check CUDA is available:
nvidia-smi

# In run-server.sh, increase GPU layers:
-ngl 999  # Offload all layers
```

### Model Not Found

Download manually:

```bash
pip install huggingface_hub
hf download unsloth/Qwen3-Coder-Next-GGUF:Qwen3-Coder-Next-UD-Q4_K_XL --local-dir models/
```

## Contributing

Both llama.cpp and quant.cpp are open-source projects:

- **llama.cpp**: [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **quant.cpp**: [github.com/quantumaikr/quant.cpp](https://github.com/quantumaikr/quant.cpp)

## License

- **llama.cpp**: MIT License
- **quant.cpp**: Apache 2.0 License

## References

- [TurboQuant Paper](https://arxiv.org/abs/2504.19874) — KV compression algorithm
- [PolarQuant Paper](https://arxiv.org/abs/2502.02617) — Polar coordinate quantization
- [QJL Paper](https://arxiv.org/abs/2406.03482) — Quantized Johnson-Lindenstrauss
- [llama.cpp Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) — Philosophy & design

## Quick Reference

| Task | Command |
|------|---------|
| Build | `bash build.sh` |
| Start server | `bash run-server.sh` |
| Test API | `bash cli-test.sh` |
| Interactive chat | `./llama.cpp/build/bin/llama-cli -m models/model.gguf` |
| Benchmark | `./llama.cpp/build/bin/llama-bench -m models/model.gguf` |
| Measure quality | `./llama.cpp/build/bin/llama-perplexity -m models/model.gguf -f input.txt` |
