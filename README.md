# SparseMiniCpm

## 依赖：InfLLM V2 稀疏注意力 CUDA 内核

本项目通过 git submodule 引入 [OpenBMB/infllmv2_cuda_impl](https://github.com/OpenBMB/infllmv2_cuda_impl)（InfLLM V2 两阶段稀疏注意力的 CUDA 实现），位于 `third_party/infllmv2_cuda_impl`，用于后续基于该稀疏化方法的训练。

### 克隆本仓库（含子模块）

```bash
git clone --recursive <this-repo-url>
# 如果已经 clone 过，忘记加 --recursive：
git submodule update --init --recursive
```

### 编译安装内核

需要 PyTorch、CUDA 11.6+ 开发工具链（nvcc）、Ninja，且有可用 GPU。若 `nvcc` 不在 `PATH` 里，先设置好 `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` 指向本机的 CUDA Toolkit 安装目录。

> **已知问题**：截至本文档编写时，`infllmv2_cuda_impl` 主分支引用的嵌套子模块 `csrc/cutlass`（fork 自 `zhangyan-didu/cutlass`，分支 `infllmv2-cuda13`）被 pin 在的 commit（`424c5a03`）中，`include/cutlass/cuda_host_adapter.hpp` 有两处 `#if` 误写成 `if`（缺少 `#`），在 CUDA 12.x 下编译会报 `#else after #else` / `#endif without #if` 而失败。该分支后续提交 `e8beb151`（"enable cuda 13.0"）已修复此问题，但上游 gitlink 尚未跟进。由于我们没有 OpenBMB 仓库的写权限，无法把修复后的提交固化进 submodule 指针，因此每次初始化子模块后需要手动切一次：
>
> ```bash
> cd third_party/infllmv2_cuda_impl/csrc/cutlass
> git checkout e8beb151
> cd ../../../..
> ```

```bash
pip install ninja
cd third_party/infllmv2_cuda_impl
pip install -e . --no-build-isolation
```

安装完成后即可在 Python 中 `import infllm_v2` 使用其 Stage1（Top-K 选块）与 Stage2（稀疏注意力前向/反向）内核。

### 更新子模块到上游最新版本

```bash
cd third_party/infllmv2_cuda_impl
git fetch origin
git checkout origin/main   # 或指定的 commit/tag
cd ../..
git add third_party/infllmv2_cuda_impl
git commit -m "Bump infllmv2_cuda_impl submodule"
```
