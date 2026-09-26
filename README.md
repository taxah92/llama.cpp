# llama.cpp

> [!IMPORTANT]
> **This is the PrismML fork of llama.cpp**, the main line behind the [Bonsai](https://huggingface.co/collections/prism-ml/bonsai) models (branch `prism`, developed as `prism-v7`). It tracks current mainline llama.cpp and adds the fork's low-bit formats and runtime features on top.
>
> **New here? Start with the [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) repo.** It downloads the right models and the correct prebuilt binaries for your hardware/backend automatically.
>
> **Which ternary model file to use:**
>
> - `*-PQ2_0.gguf` (fork group-128, ggml id 142): preferred on Metal, CUDA, HIP and CPU. About 6% smaller than group-64.
> - `*-Q2_0_g64.gguf` / 27B `*-Q2_g64.gguf` (official group-64, ggml id 42): runs on every backend here AND on mainline llama.cpp. If unsure, use this. Newer model releases name this file plain `*-Q2_0.gguf`.
> - `*-Q2_0.gguf` on OLDER model repos is the **deprecated legacy format** (group 128 stored as id 42). It does not load on these builds; the error tells you which file to get instead. If you must run it, use the frozen [`prism-v5`](https://github.com/PrismML-Eng/llama.cpp/tree/prism-v5) line and its final release [`prism-b9601`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b9601-68faa14).
>
> **Speculative decoding (dspark)** is supported via mainline's draft-dspark plus fork patches. Drafters published for older model releases need a one-time conversion with `gguf-dspark-to-dflash` (see [SPECULATIVE.md](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SPECULATIVE.md) in Bonsai-demo); newer releases ship ready-to-use drafters.
>
> Do NOT build from `prism-v6` (stale mid-migration snapshot) and do NOT mix this fork's `ggml-*` libraries with a stock llama.cpp build.

> [!NOTE]
> **taxah92 Fork: NVIDIA Volta (SM70 / Tesla V100) & MTP Optimizations**
>
> This fork ([taxah92/llama.cpp](https://github.com/taxah92/llama.cpp)) builds upon [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (branch `prism`) with fixes and performance work specifically targeting **NVIDIA Volta (SM70, Tesla V100)** and **MTP (Multi-Token Prediction)** speculative decoding. All numbers below were measured on a V100-SXM2-16GB as interleaved A/B runs in a single session (repeatability within a session σ ≈ 0.1%; note that clocks drift ~10% between sessions, so only same-session comparisons are meaningful).
>
> 1. **Fast `Q4_0` → `F16` KV dequantization (`ggml-cuda/convert.cu`)** — the main win:
>    - **Why it matters.** With a quantized KV cache the flash-attention kernels need the whole context in F16, so every decode step converts the entire KV of every layer. At 100k tokens that is ~205 MB of F16 written per layer, i.e. ~3.5 GB of writes per step: $O(\text{context})$ work per generated token, and the single largest item in the decode step.
>    - **Root cause (measured, not guessed).** The old path ran `dequantize_block_q4_0` as `<<<nb, 32>>>`: one warp per CTA, 8 elements per thread, eight scalar 2-byte stores, and the lanes of a single store instruction sit 4 bytes apart — filling only 1/16 of each 32-byte sector. The kernel was limited by store/LSU instructions and partial sectors, **not** by memory bandwidth: 301 GB/s.
>    - **Fix.** A specialized contiguous kernel writes exactly 16 bytes (8 F16 values) per thread with a single `STG.128`, 256 threads per CTA, so lanes are consecutive and one warp fills 512 contiguous bytes. `qs` is read with eight `LDG.U8`, because a Q4_0 block is 18 bytes and only 2-byte aligned (wider loads fault with `misaligned address`, and type-punning `qs` lets nvcc merge 16-bit loads into a misaligned 32-bit one). The value layout is bit-identical to the old kernel, and the non-contiguous (`kv_view=1`) path got the same treatment. Host-side guards fall back to the old kernel on shapes or alignments that do not fit.
>    - **Measured.** Conversion per layer at `kv=100096`: 1745.6 → **685.7 µs (−61%)**; non-contiguous path −35%. End to end with the model: **steps/s +23.5% at 100k** (11.31 → 13.98) and **+33% at 240k** (6.03 → 8.02), prefill +0.6%.
>
> 2. **MTP draft sampling stays on the CPU, with upstream's stochastic top-k (`common/speculative.cpp`)**:
>    - **Root cause fixed.** GPU backend offload (`backend_sampling`) bypasses construction of the CPU candidate chain, so candidate probabilities remain at $p = 0.0$. `draft()` then reads an unsorted array and emits a garbage token (e.g. `165552 ("ansir")`) on every step, collapsing draft acceptance to **0.00%**. Sampling is therefore forced onto the CPU (`this->params.backend_sampling = false`).
>    - **Top-k deliberately left at upstream's 10, not greedy.** Greedy drafts (`sparams.top_k = 1`) were tried and reverted: decode throughput dropped by **~22%** because draft acceptance falls from ~72% to ~50%. Stochastic top-k=10 is both faster and closer to upstream behaviour.
>    - **Measured** with `--spec-draft-n-max 2`: draft acceptance 46–53%, mean accepted length 1.8–2.0 tokens per step.
>
> 3. **GQA head packing in the tile kernel at any KV length (`ggml-cuda/fattn-tile.cuh`)**:
>    - Packing all query heads that share a KV head into one CTA is no longer restricted to KV lengths divisible by 256. It is neutral for this model (the KV length is always padded to a multiple of 256), but unpadded lengths get faster: at `kv=100000` one attention op goes 2252 → 1775 µs.
>
> 4. **Full 256K context on 16GB VRAM (Tesla V100-SXM2-16GB)**:
>    - Verified stable 262,144-token context in **~15.4 GiB of 16.1 GiB** VRAM on `Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf`, using a `Q4_0` KV cache with the mean-centering bias file. `LLAMA_ATTN_ROT_DISABLE=1` is **mandatory** with that bias file (it was calibrated without K-cache rotation; without the flag model init aborts).
>    - `-ub 1024` would give roughly 8% more prefill but does **not** fit at 256k even with ~940 MiB free: the VMM pool cannot assemble a contiguous block and startup aborts with `CUDA error: out of memory`. Use `-ub 768`.
>
> 5. **Multimodal projector (`mmproj`) sizing guidelines for 16GB VRAM**:
>    - **GPU Vision Offload (Fast, 281 prompt tok/s):** Set `--ctx-size 220000` with `--mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (uses 14,817 MiB VRAM, 1.57 GB headroom for image activation buffers). 512x512 image processes in 1.13s.
>    - **Max Context (256K / 262,144 tokens):** Use `--no-mmproj-offload` to keep `mmproj` in host RAM (28 GB available), keeping full 256K context in VRAM with vision processing on CPU (~4.1s per image).
>
> **Tried and reverted** (listed so nobody repeats them — each was measured):
>
> | change | measured result |
> | :--- | :--- |
> | Volta MMA shared-memory configs (`fattn-mma-f16.cuh`) | prefill 430 → 362 tok/s (**−19%**); upstream's Ampere configs are faster on V100 |
> | Forcing the vector FA kernel for quantized KV on Volta | steps/s 12.8 → 10.5 (**−18%**): the VEC kernel re-reads K/V once per query head because it does not pack heads. It is also never selected for GQA=6 anyway — selection requires $Q_{ne[1]} \cdot \text{gqa} \le 2$, and the minimum is 6 |
> | Greedy MTP drafts (`top_k = 1`) | decode **−22%** |
> | Reading `Q4_0` directly inside the tile kernel (skipping the F16 mirror) | 1888 vs 1593 µs per layer: in-kernel dequantization is issue-bound (~30 GB/s effective) and costs more than the one-off conversion |
> | Bounding `#pragma unroll` in `flash_attn_tile_iter_KQ` | within run-to-run noise; the tile kernel is ILP-bound and prefers the full unroll |
> | MMVQ launch-config sweep for sm_70 (`nwarps` × `rows_per_block`, 6 points) | every deviation from the generic config (4 warps, 2 rows) was **19–20% worse** |
> | `VDR_PQ2_0 = 2` (two 32-value chunks per `vec_dot` call) | **−25%** (44.1 → 58.4 µs); upstream's "one chunk at a time for parallelism" is the right choice |
> | Streaming stores (`st.global.wt`) for the F16 mirror | neutral (75.9 vs 75.8 µs per matmul) |
>
> **Where the remaining decode time goes** (own instrumentation, 100k context, clean decode step): `MUL_MAT` 48%, `FLASH_ATTN_EXT` 43%, everything else 9%. The step is GPU-bound — 63.9 ms of GPU time vs 17.9 ms of host time per graph. The GEMV path is `mul_mat_vec_q<ncols=2>` and runs at ~419 GB/s (47% of the V100's peak); the measured headroom is ~1.27x, and its launch parameters are already optimal.
>
> **Diagnostics.** Perf/eval cases for the paths this model actually takes (KV view variants, `kv=100096` padded lengths, permuted layouts, real `PQ2_0` weight shapes at n=1 and n=2) live in `tests/test-backend-ops.cpp`. A CUDA-event based per-operator timer is available through `GGML_OP_TIMING=1` (see `ggml-cuda/optiming.cuh`); it is inert unless the env var is set and requires `GGML_CUDA_DISABLE_GRAPHS=1`. Note that `ncu` against the running server deadlocks at the prefill→decode transition, so use this instrument instead.
>
> **Inspiration & Engineering Reference:**
> - [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) — deep engineering reference for low-bit KV cache and attention optimizations on NVIDIA Volta (SM70, Tesla V100).

---

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
