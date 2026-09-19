# ComfyUI-Hyperflow

[English README](README.md)

ComfyUI 节点包,用于运行 [HyperFlow](https://github.com/Video-Rebirth/hyperflow)——Video Rebirth 发布的 MiniMax-H3 视频+音频 8 步 LoRA。本包将其移植到 ComfyUI 原生的 MiniMax-H3 模型上,不修改任何 ComfyUI 核心文件,也**不需要自定义采样器**。

> **说明——纯 LoRA 构建(无需本节点)**:适配器同时以提取转换后的、兼容官方 Load LoRA 节点的形式发布在 [HF 仓库](https://huggingface.co/drbaph/Hyperflow-Comfyui)——`minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors`(3.93 GB)/ `..._pruned_bf16.safetensors`(3.91 GB),以及 rank-20 压缩变体(约 318/316 MB)。这些仅应用**主干 LoRA**——双时间 `(t, r)` 条件机制是本节点独有的,没有它输出会偏离官方发布的模型。

该适配器在官方基座之上包含两部分:一个 LoRA(rank 256,不合并的 bf16 分支),以及**双时间 `(t, r)` 条件机制**——每一步都以其正在积分的区间作为条件,`r = 1 - sigma_next`。本包只替换原生模型中的一个调用(`time_embedder(t_vals)`),将其换成混合嵌入,并通过 ComfyUI 自身的权重适配机制应用 LoRA(默认 bypass / 可选 merge)。因此,量化基座、模型卸载以及内存可视化器的表现与任何普通 MODEL 补丁完全一致。

## 安装

1. 克隆到 `ComfyUI/custom_nodes/ComfyUI-Hyperflow`。
2. 从 [drbaph/Hyperflow-Comfyui](https://huggingface.co/drbaph/Hyperflow-Comfyui) Hugging Face 仓库下载**其中一个**转换后的权重文件(或直接打开节点的 `download_if_missing` 开关,首次运行时会自动下载):

```
📂 ComfyUI/
└── 📂 models/
    └── 📂 hyperflow/
        ├── custom_node_hyperflow_8step_v1.0_comfyui.safetensors         (3.67 GiB,完整基座——官方发布的 8 步模型)
        └── custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors  (3.64 GiB,剪枝/曲线基座——仅主干,单时间运行)
```

   `hyperflow.json` 清单随本包一起发布在 `assets/` 目录中——用户只需下载 `.safetensors` 文件。转换后的文件保留了 HyperFlow 完整的自描述文件头(gate、sigma 网格、rank);原始 diffusers 格式的文件会被明确拒绝,加载时绝不进行任何转换。

3. 重启 ComfyUI。

## 使用方法

```
Load Diffusion Model (MiniMax-H3)
  -> ApplyHyperFlow            (MODEL -> MODEL + SIGMAS)
  -> [可选] Model Attention Backend        (核心节点;稠密注意力后端,含 comfy-kitchen int8)
  -> [可选] Model Sparse Attention         (核心节点;sol-attn / sla / vsa)
  -> SamplerCustomAdvanced + guider + Euler    (接入 SIGMAS 输出,而不是调度器)
```

- `ApplyHyperFlow` 会将训练好的 9 点 sigma 网格作为 **SIGMAS** 输出——请将它接入 `SamplerCustomAdvanced` 以替代 `BasicScheduler`。任何核心采样器/引导器都可以使用;双时间端点由 `sample_sigmas` 推导(与原生 H3 最终层使用的机制相同)。
- `lora_mode`:`bypass`(默认)在运行时应用 LoRA——与参考实现的未合并 bf16 分支一致;`merge` 将 LoRA 折叠进权重——显存占用最低,但在量化基座上会变软。
- `download_if_missing`:从 [drbaph/Hyperflow-Comfyui](https://huggingface.co/drbaph/Hyperflow-Comfyui) 下载所选 `variant`(`auto` 自动匹配检测到的基座)到 `models/hyperflow/`——仅下载发布的 `.safetensors` 文件,别无其他。默认关闭。
- `ApplyHyperFlowAdvanced` 提供 gate 与 sigma 网格覆盖选项(用于消融实验;默认值可精确复现官方发布的模型)。

### Sol-Attn(可选稀疏注意力)——核心节点设置

HyperFlow 官方验证的 Sol-Attn 配方对应核心 **Model Sparse Attention** 节点:

| HyperFlow(参考实现) | Model Sparse Attention |
| --- | --- |
| `dense_steps = 2`(共 8 步) | `start_percent = 0.25` |
| `dense_layers = (0, 1)` | `dense_blocks = "0,1"` |
| `tau = 1.0` | `tau = 1.0`(方法选 `sol-attn`) |
| sink tokens: 无 | 保持默认 `exact_kv_and_rows`(对条件行严格更精确) |
| — | `extra_tokens = 0` 最接近官方配方(256 = 更多质量冗余) |

SLA 是另一种不同的稀疏方法——仅建议与针对 SLA 训练的权重配合使用。

## 说明

- **基座自动检测**:节点会自动检查加载的模型——完整基座(有 `time_embedder`)或剪枝/曲线基座(无 `time_embedder`)——并强制要求匹配的权重构建,不匹配时会明确报错并指出正确的文件。剪枝基座构建仅应用主干 LoRA,以单时间运行(非官方配方)。
- **量化基座**(int8/融合算子):被基座折叠进融合内核、无法挂钩的 LoRA 目标会被自动检测并改经 merge 路径应用——控制台报告中会列出 `N fused/int8 targets via merge`。
- **模型采样 shift**:H3 模型默认就是视频 shift 12 / 音频 shift 3。核心 ModelSampling 节点放在 `ApplyHyperFlow` 之后,只有需要非默认 shift 时才必须添加。
- **aimdo malloc-graph**:在模型编译器会因打过补丁的 MiniMax-H3 前向而崩溃的 Comfy 版本上,本节点仅对自身模型调用禁用编译器(与 VDN-H3 节点相同的解决方案)。
- 权重文件是 MiniMax-H3 的模型衍生作品,遵循 [MiniMax H3 社区许可协议](https://huggingface.co/videorebirth/hyperflow);本包代码为 Apache-2.0(schedule/embedder 移植部分衍生自 HyperFlow 与 diffusers 代码,参见上游 `THIRD_PARTY_NOTICES.md`)。

## 测试

纯 CPU 测试,无需任何模型文件(使用 ComfyUI 的 venv 运行):

```
python -m pytest tests/ -q
```

## 致谢

- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)(GitHub):基座模型、VAE、条件器与官方工作流。
- [AnyFlow](https://github.com/NVlabs/AnyFlow)(GitHub;Gu 等,2026):双时间 `(t, r)` 条件机制背后的流图(flow-map)公式。
- [Sol-Attn](https://github.com/NVlabs/Sol-Attn)(GitHub;Li 等,2026):可选的注意力内核。

感谢以上项目的作者。
