# ComfyUI-Hyperflow

[English README](README.md)

<img width="1130" height="476" alt="image" src="https://github.com/user-attachments/assets/327cbbae-a2d7-4f86-b7ec-4dd12b6ce9e9" />

ComfyUI 节点包,用于运行 [HyperFlow](https://github.com/Video-Rebirth/hyperflow)——Video Rebirth 发布的 MiniMax-H3 视频+音频 8 步 LoRA。本包将其移植到 ComfyUI 原生的 MiniMax-H3 模型上,不修改任何 ComfyUI 核心文件,也**不需要自定义采样器**。

> **说明——不想装本节点?** 可以直接使用**独立 LoRA 构建**——已提取并转换为纯 ComfyUI 格式,用**官方自带的 `Load LoRA` 节点**即可加载,无需安装任何东西:[drbaph/MiniMax-H3-Turbo-Lora-ComfyUI](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/)——`minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors`(3.93 GB)/ `..._pruned_bf16.safetensors`(3.91 GB),以及 rank-20 压缩变体(约 318/316 MB)。这些仅应用**主干 LoRA**——双时间 `(t, r)` 条件机制是本节点独有的,没有它输出会偏离官方发布的模型。

该适配器在官方基座之上包含两部分:一个 LoRA(rank 256,不合并的 bf16 分支),以及**双时间 `(t, r)` 条件机制**——每一步都以其正在积分的区间作为条件,`r = 1 - sigma_next`。本包混合两个时间嵌入,并通过 ModelPatcher 在原生模块和输出头之间保留不同的 `(t, r)` 行,包括视频和音频共享 `t = 0` 的第一步。无需修改 ComfyUI 核心文件。

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
- `lora_mode`:`bypass`(默认)在运行时应用 LoRA——与参考实现的未合并 bf16 分支一致;`merge` 将主干 LoRA 合并进权重,但两个时间投影仍使用 bypass,避免污染端点分支的原始权重。量化权重合并可能改变数值结果。
- `download_if_missing`:从 [drbaph/Hyperflow-Comfyui](https://huggingface.co/drbaph/Hyperflow-Comfyui) 下载所选 `variant`(`auto` 自动匹配检测到的基座)到 `models/hyperflow/`——仅下载发布的 `.safetensors` 文件,别无其他。默认关闭。
- `ApplyHyperFlowAdvanced` 提供 gate 与 sigma 网格覆盖选项(用于消融实验;默认值可精确复现官方发布的模型)。

### Sol-Attn(可选稀疏注意力)——核心节点设置

HyperFlow 官方验证的 Sol-Attn 配方对应核心 **Model Sparse Attention** 节点:

| HyperFlow(参考实现) | Model Sparse Attention |
| --- | --- |
| `dense_steps = 2`(共 8 步) | `start_percent = 0.16` |
| `dense_layers = (0, 1)` | `dense_blocks = "0,1"` |
| `tau = 1.0` | `tau = 1.0`(方法选 `sol-attn`) |
| sink tokens: 无 | `sink_conditioning = "off"` |
| — | `extra_tokens = 0` |
| 不按序列长度禁用 | `min_tokens = 0` |

SLA 是另一种不同的稀疏方法——仅建议与针对 SLA 训练的权重配合使用。

### 剪枝/曲线基座——实验性曲线重拟合

剪枝版 MiniMax-H3 基座没有 `time_embedder`,双时间 `(t, r)` 路径无法应用——默认情况下剪枝构建只应用主干 LoRA、以单时间运行(非官方配方)。**实验性曲线重拟合**通过一个绑定到检查点的微型拟合文件(约 34 KB,已内置 `assets/curve_fits/`,覆盖 FL2VA 与 REF2VA 剪枝 int8 检查点——无需下载、不增加任何模型权重)恢复大部分双时间信号。

**剪枝基座推荐设置:**

- 开启 **`experimental_curve_refit`**(两个 Apply 节点上都有;若被隐藏,右键节点 → *显示可选控件*),**`strength` 保持 `1.0`**,gate/σ 覆盖留空。匹配为分级机制:与拟合文件完全一致时静默应用;基座或适配器字节不同但文件名一致(镜像、HF 下载副本)时以**最佳 effort 模式应用并在控制台给出警告**——配方(strength/gate/σ)始终会被校验;未知检查点或 MODEL 被其他节点修改时才回退到仅主干模式,并在日志中输出文件哈希便于排查。
- `lora_mode` 用 **`bypass`**,最接近参考分支的行为。
- 采样与完整基座相同:**Euler** + 节点的 **SIGMAS** 输出 + 上方的 Sol-Attn 参数表。
- 控制台出现 `[hyperflow] curve refit disabled: <原因>` 表示已回退——请检查 strength/覆盖/检查点。

相对完整基座实测(固定提示词/种子/初始潜变量):FL2VA 视频余弦相似度 **0.81 → 0.88**,REF2VA 音频 **0.93 → 0.99**。它恢复了*大部分*而非全部缺失条件;完整基座仍是参照标准。详细数据见 [`validation/curve_findings.md`](validation/curve_findings.md)。新的剪枝检查点需要生成自己的拟合文件(`validation/curve_spike.py`)。

## 说明

- 旧节点权重的 SwiGLU 行顺序会在加载时自动修正,无需重新下载。新转换文件带有 `hyperflow_fc1_layout=gate_value`,不会重复交换。
- 默认网格和 shift 下,`start_percent = 0.16` 保持前两步使用密集注意力;该百分比不是步数比例。

- **基座自动检测**:节点会自动检查加载的模型——完整基座(有 `time_embedder`)或剪枝/曲线基座(无 `time_embedder`)——并强制要求匹配的权重构建,不匹配时会明确报错并指出正确的文件。剪枝基座构建仅应用主干 LoRA,以单时间运行(非官方配方)。
- **剪枝/曲线基座的实验性曲线重拟合**:详见上方专门章节。
- **量化基座**(int8/融合算子):被基座折叠进融合内核、无法挂钩的 LoRA 目标会被自动检测并改经 merge 路径应用——控制台报告中会列出 `N fused/int8 targets via merge`。
- **模型采样 shift**:H3 模型默认就是视频 shift 12 / 音频 shift 3。核心 ModelSampling 节点放在 `ApplyHyperFlow` 之后,只有需要非默认 shift 时才必须添加。
- **aimdo malloc-graph**:在模型编译器会因打过补丁的 MiniMax-H3 前向而崩溃的 Comfy 版本上,本节点仅对自身模型调用禁用编译器。
- 权重文件是 MiniMax-H3 的模型衍生作品,遵循 [MiniMax H3 社区许可协议](https://huggingface.co/videorebirth/hyperflow);本包代码为 Apache-2.0(schedule/embedder 移植部分衍生自 HyperFlow 与 diffusers 代码,参见上游 `THIRD_PARTY_NOTICES.md`)。

## 更新 ComfyUI 后停在 “Model Initializing”

该状态覆盖整个第一步采样,包括延迟加载权重和合并 LoRA,并不代表一定是编译器卡死。`merge` 减少常驻适配器显存,但合并基座权重仍需要临时内存和计算时间。

相比 v0.36,ComfyUI v0.37 新增了自动检测快速磁盘。若更新后卡住,可用 `--disable-fast-disk` 重启 ComfyUI,重试同一工作流以排查原因。这是诊断步骤,不是已确认的 HyperFlow 修复。若仍卡住,请提供 ComfyUI commit、完整控制台日志、基座及 LoRA 文件名、GPU/显存和系统内存,并说明发生在第一块还是后续块。若日志包含 `[hyperflow] ... applied`,请一并提供。

## 致谢

- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)(GitHub):基座模型、VAE、条件器与官方工作流。
- [AnyFlow](https://github.com/NVlabs/AnyFlow)(GitHub;Gu 等,2026):双时间 `(t, r)` 条件机制背后的流图(flow-map)公式。
- [Sol-Attn](https://github.com/NVlabs/Sol-Attn)(GitHub;Li 等,2026):可选的注意力内核。

感谢以上项目的作者。
