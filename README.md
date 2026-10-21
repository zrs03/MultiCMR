# MultiCMR：SAX 时空邻域输入

当前版本 `sax_spatiotemporal_25d_v18` 基于原有残差 CNN、两阶段解剖交互与 FiLM，采用 CardioState 的 SAX 输入组织。不是完整复现 CardioState：保留原 backbone、跨视图交互和损失；2CH/4CH 仍使用原五帧时间窗。

## 输入和输出

| 视图 | 缓存输入                  | 训练标签           | 网络输出      |
| ---- | ------------------------- | ------------------ | ------------- |
| 2CH  | `[K,H,W]`，默认 K=5     | `[H,W]` 中心帧   | `[B,3,H,W]` |
| 4CH  | `[K,H,W]`，默认 K=5     | `[H,W]` 中心帧   | `[B,6,H,W]` |
| SAX  | `[5,H,W]`，固定通道含义 | `[H,W]` 中心切片 | `[B,4,H,W]` |

SAX 的五个通道严格依次为：

```text
I(t,z), I(t-1,z), I(t+1,z), I(t,z-1), I(t,z+1)
```

输入 `[B,1,5,H,W]` 在第一层二维卷积前合并为 `[B,5,H,W]`。SAX 使用独立的二维 encoder、主 decoder 和辅助 decoder，三个分割头均为二维。长轴 encoder 独立，主/辅助 decoder 仍在两长轴视图之间共享。

三次空间下采样与 patch size=1 不变。128 输入时，各视图均有 `16×16=256` 个空间 patch tokens，加五个解剖 tokens 共 **773**。五个邻域图像是通道，不是五倍 token。

2CH/4CH patches 先直接交互，再与对应解剖 tokens 双向交互；SAX 与其关联的三个解剖 tokens 双向交互。最终解剖 tokens 经 FiLM 调制三视图主 decoder。长轴独立空间适配层、模态掩码、随机 token key/value 屏蔽保持不变。

## SAX 排列与预处理

默认从 `DATA_ROOT`（默认 `rawdata/CINE_MULTI`）按划分读取相位数 T：train 使用 `sax_slice_info.json`，val 使用 `id_slice_info_valid.json`，test 使用 `sax_slice_info_test.json`。不再默认使用仅含训练病例的 `rawdata/id_mapping.json`。显式路径优先级为：划分专属覆盖 > 通用覆盖 > 数据根目录下的默认文件；缺失文件不会自动回退到其他划分。采用 **k=z*T+t**：一个空间切片内的所有相位连续排列。原始 `[H,W,Z*T]` 保留全部帧，只缩放 H/W，绝不沿混合轴插值。总帧数必须能被 T 整除，缺少病例、非正整数相位数或整除失败会报错，不猜测。

SAX 时间与空间边界均用反射；长度为 1 时重复自身。反射分别在 T 和 Z 上执行，时间邻居不会跨到另一切片，空间邻居不会跨越心底/心尖。长轴仍保留原 edge 时间窗边界规则。

```bash
python -m pip install -r requirements.txt
bash run.sh preprocess
```

默认生成：

- `preprocess/matched_sax_context.yaml`，schema=`sax_context_v2`。
- `preprocess/processed_sax_context/{split}/{case_id}/`。
- `sa_full.npz`：全部原生帧，布局 FHW，记录 T、Z 与 `slice_phase` 排列。
- `sa_tXXX_zXXX.npz`：五通道上下文、中心标签、邻居索引。
- 长轴完整序列和时间窗缓存；同一长轴时间窗在不同 SAX 切片样本间复用。

每病例生成 `max(T_2ch,T_4ch,T_SAX)×Z` 个配对样本。三个视图按归一化序列位置最近邻匹配；这只是近似相位配对，不保证不同采集间生理同步。先沿用患者划分再拆样本，禁止同一患者的切片/相位泄漏到不同划分。

自定义参数：

```bash
SAX_PHASE_INFO=/path/to/phases.json bash run.sh preprocess
SAX_PHASE_INFO_TRAIN=/path/to/train.json SAX_PHASE_INFO_VAL=/path/to/val.json SAX_PHASE_INFO_TEST=/path/to/test.json bash run.sh preprocess
LAX_SIZE="5 128 128" SAX_SIZE="5 128 128" bash run.sh preprocess
```

各 JSON 为 `{"001":25,"002":30,...}`。首位 SAX 尺寸必须是 5，不再是空间深度。切片数、相位数由元数据和原始形状决定。旧 `128³` 缓存已混合插值，不能转换回准确邻域，必须从原始 NIfTI 重新预处理。

## 训练

```bash
# 建议显式增加每患者上下文采样量：实验起点，非已验证最优值
FP16=1 WINDOWS_PER_PATIENT=32 bash run.sh train-multi-gpu
# 单 GPU
FP16=1 WINDOWS_PER_PATIENT=32 bash run.sh train
```

默认输出目录 `checkpoints_sax_context`。沿用 `WINDOWS_PER_PATIENT=4` 默认值，但其单位现在是“一个 SAX 中心切片/相位 + 两个长轴时间窗”。它不再监督一整个 SAX 体积。建议从 32 做对照，0 为全部样本。比较总训练步数与病例覆盖，不能把相同 epoch 数当作相同预算。验证遍历全部配对样本，日志仍为样本加权指标，不是病例级指标。

训练均匀抽取七种非空模态组合；缺失视图不编码、不作为 key/value、不计算监督。验证使用完整模态。`TOKEN_DROP_RATE` 默认 .05，针对至少两个存在视图的样本，只屏蔽解剖 tokens 读取部分 patch key/value；不屏蔽输出 queries，验证关闭。

保留空间翻转与小幅仿射：同一上下文五通道使用相同变换，中心标签同步变换；不翻转邻域通道轴。未新增强度增强、EMA 或其他论文组件。

AdamW 默认 lr=1e-4、weight_decay=1e-4，ReduceLROnPlateau 调度器不变。FP16 用 CUDA autocast/GradScaler，损失与指标仍使用 FP32。

## 损失和 checkpoint

`utils/losses.py` 是统一入口。每个有效样本—视图：

`Dice+CE + 0.05*boundary + 0.2*auxiliary(Dice+CE) + 0.1*deep(Dice+CE)`。

三个视图现在均对中心二维标签计算损失。Dice 包含背景；边界只计算前景。主 decoder 的半分辨率 d2 head 提供单尺度深度监督；独立辅助 decoder 是另一条监督路径。验证不计算辅助或 deep loss。`DEEP_WEIGHT=0` 可关闭 deep loss。

**旧 v14、Swin UNETR 或其他旧 checkpoint 均不兼容，需重新训练。** 新版本支持完整 resume：

```bash
FP16=1 WINDOWS_PER_PATIENT=32 RESUME=checkpoints_sax_context/last_model_3d_multihead.pth EPOCHS=300 bash run.sh train-multi-gpu
```

文件名为兼容启动脚本仍保留 `*_3d_multihead.pth`，不代表 SAX 仍使用三维网络。resume 恢复模型、AdamW、调度器、GradScaler、epoch 和随机数状态；EPOCHS 是目标总轮数，命令行 LR 不覆盖恢复后的学习率。

## 完整病例推理与评估

```bash
FP16=1 bash run.sh test
FP16=1 EVAL_MODE=all-seven bash run.sh test
```

推理从完整缓存的 T、Z 构造与训练完全相同的上下文。遍历各相位、各空间切片，只预测对应中心。SAX-only 仍需遍历 Z*T 个中心，不再只运行一次网络。长轴输出对不同 SAX 切片上下文产生的重复预测平均概率；SAX 只在同一个 `(t,z)` 内平均重复预测，绝不跨相位平均。

SAX 按 `k=z*T+t` 恢复 `[Z*T,H,W]`，导出 `[H,W,Z*T]`；长轴导出 `[H,W,1,T_native]`。当前仍为模型平面分辨率和单位 affine，不是原始物理空间。SAX 评估按 `t::T` 组成空间体积，再逐相位计算 Dice/IoU/HD95 并平均，避免对时间维计算空间距离。HD95 仍为模型网格单位，不是毫米；不能与旧混合堆叠指标直接比较。EF 工具要求原始空间恢复，不能直接使用当前模型空间导出计算临床量。

测试：

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

覆盖邻域索引、反射边界、混合轴不插值、预处理/推理一致性、各相位重建、模态掩码、监督梯度和断点恢复。
