# GH200 Strata ARM64 Port (experimental)

Target: NVIDIA Grace Hopper GH200, Grace aarch64 CPU, 96 GB HBM GPU, Grace CPU RAM; Qwen3.8-Flash-Next (Swift-1.5 family); eight concurrent Qwen Code clients with 262144 token *per-request ceiling*, xhigh thinking, aspirational 200 total output tokens/second.

This tree contains an experimental ARM64 CPU path adapted from [upstream PR #409](https://github.com/Niko1221/Strata/pull/409) and then rebased manually onto the current engine. **No GH200 end-to-end inference measurement has been made here.**

## Implemented

- Separate aarch64 CPU kernel source, native GGUF expert path through ggml-cpu NEON and architecture-independent spin/fence wrappers.
- Source compilation selection for ARM64, CUDA sm_90 with an available CUDA toolkit.
- Reuse built-in 8-slot batched inference, built-in per-slot KV state, 256K context and xhigh chat template. No invented parallel execution engine.
- Optional helper to create an 8-slot/256K profile without modifying the installed config; KV in Grace RAM through the existing --kv-resident mechanism.
- CI source smoke compile and reproducible API throughput script.

## Not yet validated

- Successful full CUDA binary link and real-model decode on GH200.
- Eight fully populated 256K histories concurrently; 8 slots allocated does not mean eight *full* histories fit.
- Stability and OpenBMC issue correctness with eight agents.
- 200 tok/s aggregate. This is a **target**, not a measured result.

## Hardware inspection

Run directly on the GH200 Linux machine:

~~~sh
uname -m
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
free -h
nvcc --version
~~~

Expected: aarch64, Hopper compute capability 9.0, and a dedicated HBM allocation. Unlike DGX Spark GB10, GH200 Grace RAM and GPU HBM are **not one physically interchangeable VRAM pool**; the coherent interconnect still has finite bandwidth and capacity.

## Build & install

Choose the Swift-1.5 Flash-Next family for long reasoning, not the separate 27B model. Choose IQ3_XXS when RAM permits, or IQ2_XS for easier first bring-up:

~~~sh
git checkout feat/gh200-arm64-inference
python3 -m py_compile setup.py gh200/configure.py gh200/benchmark.py
python3 -m unittest discover -s gh200 -p 'test_*.py'
./setup.sh --setup --yes --family swift --model IQ3_XXS \
  --context 262144 --parallel 8 --kv-streaming on \
  --vision no --build --no-start
~~~

Manual engine-only CMake example (assuming the pinned llama.cpp checkout is ready):

~~~sh
cmake -S . -B build-gh200 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DSTRATA_BUILD_TESTS=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16 \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp
cmake --build build-gh200 --target strata -j 8
~~~

The ARM ISA extension flags depend on the machine's /proc/cpuinfo features; setup.py derives them. Do not assume all extensions exist. Check model disk storage; a large GGUF and PLE index require ample NVMe and LPDDR.

After installing the model, generate a separate opt-in profile (adapt file paths to the actual setup output):

~~~sh
python3 gh200/configure.py /path/to/strata-swift-iq3_xxs.json \
  --output /path/to/strata-gh200.json --slots 8 --context 262144
~~~

The generated JSON uses parallel=8, an uncapped thinking budget, KV int8 with hot 32K KV window and cold storage in Grace RAM, and expert-cache auto. The existing model config is never overwritten. If current KV is k8v4, the tool refuses that incompatible combination.

Point Strata server at this JSON file using the normal configuration invocation. Never expose the server beyond loopback without an API key. Each Qwen Code client must send reasoning_effort=xhigh; the model template also defaults to xhigh if absent. This project does not secretly replace xhigh with low/medium. Note that an existing client setting can explicitly override thinking.

Start the server and inspect the startup INFO batch_slots=N. Verify the *actual*
server allocation (exit nonzero if the server quietly reduced slots or context):

~~~sh
python3 gh200/verify_server.py --url http://127.0.0.1:8080 --slots 8 --context 262144
~~~

This checks /v1/status concurrency.serving and context.max_positions. Do not mistake
eight connected Qwen Code processes for eight simultaneously allocated slots.
Passing this check does **not** prove eight full 256K contexts fit in memory.

## Benchmark

~~~sh
python3 gh200/benchmark.py --url http://127.0.0.1:8080 \
  --slots 1 --max-tokens 512 --output gh200-single.json
python3 gh200/benchmark.py --url http://127.0.0.1:8080 \
  --slots 8 --max-tokens 512 --target 200 --output gh200-eight.json
~~~

The script sends concurrent OpenAI-compatible requests, explicitly sets xhigh, reads actual API completion_tokens and divides by end-to-end wall time. Prefill, queuing and network overhead lower this result relative to isolated decode throughput. The output reports observed tok/s and whether 200 was reached; it never fabricates a result.

Full acceptance testing additionally needs 8K/64K/128K/256K real token contexts, simultaneous 8-request OpenBMC scenarios with tool calls, correction/patch/test outcomes, memory high-water marks, pinned Grace RAM, swap, MTP acceptance, and no memory allocation errors.

## Limits and safety

If eight *active* 256K contexts exceed host pinned memory or HBM, reduce concurrently running sessions or queue work. Do not silently shorten per-slot context or thinking depth; preserve the requirement and report capacity limits. Hopper performance optimization beyond the portable bring-up is still needed for 200 tok/s.
