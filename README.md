# MultiCMR

2CH/4CH 使用以目标帧为中心的 **5 帧时间窗**，2D encoder/decoder 只预测中心帧；每个训练样本同时包含同病例 SAX 体积。使用原生 patch、5 个 anatomy tokens、两阶段 attention 和缺失模态掩码，不使用 fusion tokens，SAX 使用 3D encoder/decoder。

| 模态 | 缓存图像          | 缓存标签          | 批量输入              | 分割 logits           |
| ---- | ----------------- | ----------------- | --------------------- | --------------------- |
| 2CH  | `[5,128,128]`   | `[128,128]`     | `[B,1,5,128,128]`   | `[B,3,128,128]`     |
| 4CH  | `[5,128,128]`   | `[128,128]`     | `[B,1,5,128,128]`   | `[B,6,128,128]`     |
| SAX  | `[128,128,128]` | `[128,128,128]` | `[B,1,128,128,128]` | `[B,4,128,128,128]` |

## 预处理与样本

```bash
python -m pip install -r requirements.txt
bash run.sh preprocess
# 自定义正奇数窗口长度 K，H/W 是空间尺寸
LAX_SIZE="3 128 128" bash run.sh preprocess
```

必须重新预处理。旧 matched.yaml 和完整序列训练缓存不能直接用于窗口训练。原始长轴输入支持 `[H,W,T]` 和 `[H,W,1,T]`，保留原始帧数，仅重采样 H/W；图像使用非零区域 z-score 和双线性插值，标签使用最近邻。时间窗边界复制首尾帧，例如 K=5、中心帧 0 对应 `[0,0,0,1,2]`。不沿时间插值或重采样到 64 帧。

每病例生成 `max(T_2ch,T_4ch)` 个样本。两视图按归一化序列位置最近邻配对，中心索引为 `round(i*(T_view-1)/(N-1))`；所有原始帧均被覆盖，较短视图可能重复使用某些中心帧。这只是序列位置配对，不保证生理心动相位一致。应确认原始长轴最后一轴确实为连续时间，混合切片/多序列堆叠必须先整理。

文件写入 `preprocess/processed/{split}/{case_id}/`：

- `2ch_w0000.npz` / `4ch_w0000.npz`：窗口图像、中心帧标签、frame_indices、center_index、原始 affine/shape。
- `sa_full.npz`：每病例仅保存一份 SAX，所有窗口记录引用它。
- `2ch_full.npz` / `4ch_full.npz`：保留所有原始时间帧的空间重采样结果，用于病例级推理和序列重建。

`matched.yaml` 的 `splits` 是窗口级记录，`case_splits` 是病例级完整序列记录。先沿用原始患者 train/val/test 划分，再拆窗口，避免同一病例泄漏到不同集合。无标签 test 也会生成窗口。`--manifest-only` 已移除，训练使用离线拆分缓存。

SAX 继续使用原有预处理：最后一轴移到最前并缩放至指定 DHW，不在本次改动中推断或拆分其心动周期；使用前仍须核对原始堆叠是否具有预期语义。

## 网络与训练

```bash
bash run.sh train
NUM_GPUS=2 GPU_DEVICES=0,1 bash run.sh train-multi-gpu
```

模型接收三视图字典和 `[B,3]` 模态掩码。长轴 `[B,1,K,H,W]` 合并为 `[B,K,H,W]`，在首层 2D 卷积融合时间窗，亦兼容直接输入 `[B,K,H,W]`。`temporal_size` 表示正奇数窗口长度，默认 5。两个长轴视图各自使用 encoder、共用 2D decoder、各自使用中心帧分类头；SAX 使用独立 3D 分支。主 decoder 和辅助 decoder 使用融合窗口后的二维 skips，只监督中心帧，主 decoder 的 d2 层另有单尺度深度监督。

默认尺寸下，每个长轴窗口生成 `16×16=256` 个 patch tokens，SAX 为 4096 个，另有 5 个 anatomy tokens，总数为 **4613**。kernel=1、stride=1 的空间 patch embedding、空间位置编码、模态内 Transformer 保留；`token_grid` 仅控制 encoder 位置编码初始网格，无 fusion tokens。时间维在首层卷积融合，不再加入时间位置编码。交互后的空间 patches 插值回 CNN 瓶颈尺寸并残差注入；解剖 tokens 通过 FiLM 调制解码器。

跨视图融合使用两个顺序 cross-attention 阶段，不再构建所有 patch 的联合 self-attention：

1. **长轴阶段**：2CH/4CH patches 先做联合 self-attention，再由 5 个解剖 token 作为 query，读取 2CH/4CH patch；2CH 仅关联心肌/左室腔，4CH 关联全部 5 个。然后长轴 patch 作为 query，读取更新后的适用解剖 token。
2. **SAX 阶段**：心肌/左室腔/右室腔 token 读取 SAX patch，再由 SAX patch 读取更新后的这三个 token。心房 token 在本阶段保持不变。

`--inter-layers` 目前必须为 2，对应上述两个阶段；不再表示全局层数。每阶段两次矩形 attention，无 patch-to-patch 或 anatomy-to-anatomy self-attention。每个 cross block 使用 pre-norm、残差及 FFN。缺失视图不提供 key/value，缺失 patch 输出清零；解剖 query 没有有效图像证据时保持原值，用零 fallback 避免全屏蔽 softmax。

训练仍均匀采样 7 种非空模态掩码；每条数据包含 SAX，但随机掩码可将它屏蔽。验证使用完整模态。缺失视图不编码、不作为 attention key/value、不计入监督。长轴 Dice+CE、边界损失及指标为中心帧二维计算；SAX 为三维计算。窗口内所有帧与中心标签采用一致的空间翻转。

日志中的缓存数量是**窗口数**。训练默认每患者每轮随机抽取最多 4 个不同窗口，验证仍遍历全部窗口；SAX 在选中的窗口参与监督。训练脚本的 val Dice 是窗口加权统计（SAX 也按窗口计数），与独立 test.py 的病例级重建分数不同，帧数不同的病例权重不同。窗口抽样改变了每轮训练步数，请按日志中的 train_steps 比较总优化步数，不能仅比较 epoch 数。

## 完整序列评估

```bash
bash run.sh test
SPLIT=test bash run.sh test
EVAL_MODE=all-seven bash run.sh test
python -m unittest discover -s tests -v
```

test.py 读取病例级完整序列，构造与预处理一致的成对时间窗，每次加同一份 SAX。一个病例在同一评估场景的所有窗口固定使用相同掩码。长轴中心帧预测按原始时间顺序重建，重复中心帧先平均类别概率再 argmax；SAX 对所有窗口的类别概率取平均后 argmax，每病例只评估一次。SAX-only 场景只执行一次网络。为控制 SAX 和 attention 显存，当前推理按窗口顺序执行，完整病例比原先一次全序列 forward 更慢。

支持 full、random、both、all-seven；random 场景各视图病例子集可能不同。导出仍为模型空间 NIfTI：长轴 `[H,W,1,T_native]`，SAX `[H,W,D]`，使用单位 affine。HD95 为模型网格单位，并非毫米。`calculate_lv_ef.py` 要求原始 SAX 布局和 spacing，不能直接使用这些导出文件。

尺寸选项为 `--lax-size K H W`、`--sax-size D H W`，启动脚本对应 `LAX_SIZE`、`SAX_SIZE`。train/test 默认从 YAML/checkpoint 读取配置，改变窗口或空间尺寸须重新预处理。新 checkpoint 标记为 `center_window_lax_joint_cross_patch1_deep_v14`。旧 64 通道整序列输出、逐帧时间 token、三路 3D 等 checkpoint 均不能用于此结构，需重新训练。

## 断点续训

每轮结束后保存 `last_model_3d_multihead.pth`，验证 Dice 改善时另存 `best_model_3d_multihead.pth`。两者均包含模型、AdamW、学习率调度器、已完成 epoch、历史最佳 Dice 及各 rank 的随机数状态。使用临时文件加原子替换保存，避免写入中断破坏上一次 checkpoint。

```bash
python train.py --resume checkpoints/last_model_3d_multihead.pth --epochs 300
RESUME=checkpoints/last_model_3d_multihead.pth EPOCHS=300 bash run.sh train
RESUME=checkpoints/last_model_3d_multihead.pth EPOCHS=300 bash run.sh train-multi-gpu
```

`--epochs` / `EPOCHS` 是目标总轮数：checkpoint 完成第 100 轮时，以上命令执行第 101–300 轮。日志追加而非清空。模型结构自动采用 checkpoint 配置，数据窗口尺寸、类别和视图顺序必须匹配；优化器恢复保存的学习率，`--lr` 不会覆盖完整续训的学习率。数据路径、batch size、worker 数、损失权重等沿用当前命令，建议与原训练保持一致。

支持从 best 或 last 续训；last 通常包含最近进度。同架构 FiLM checkpoint 只有模型权重时会明确告警：加载权重，但优化器/调度器重新初始化，epoch 从 1 开始。旧 fusion-token、cross-attention 和 64 通道架构仍不兼容。更改 GPU 数量可继续训练，但随机数流无法按原 rank 对应恢复，会告警；不保证与原训练轨迹一致。续训从完整 epoch 边界开始，不恢复中断 epoch 内尚未保存的 batch。

指定新的 OUTPUT_DIR 时，新 best 文件只有超过历史最佳分数才会写入；历史最佳文件仍位于原目录。相同数据、参数、worker 和 world size 是复现连续训练轨迹的前提。

## 解剖 token FiLM 调制

两阶段 Transformer 更新 5 个解剖 tokens。主 decoder 的三个尺度使用 FiLM 替代空间 query 对解剖 tokens 的 cross-attention：`F_out = (1 + delta_gamma) * F + beta`。2CH 选取心肌/左室腔 2 个 token，4CH 使用全部 5 个，SAX 使用心肌/左室腔/右室腔 3 个。按固定解剖顺序拼接选定 token，经 LayerNorm 和 MLP 生成每通道的缩放与偏移，广播到所有空间位置。每视图、每尺度都有独立生成器；2CH/4CH 继续共享解码卷积。

FiLM 最后一层权重和偏置零初始化，因此初始为恒等映射。主分割经调制特征输出；辅助 decoder 仍不使用解剖调制。模态掩码和 patch 机制保留；fusion tokens 已移除。这是通道级全局条件调制，不保证某个 token 仅影响对应器官像素。

窗口预处理缓存可继续使用，无须重新拆分。旧无适配层/fusion-token/cross-attention checkpoint 不支持直接测试或完整 resume，需重新训练 FiLM 模型；建议指定新的 OUTPUT_DIR 避免覆盖旧实验。

## 患者窗口采样与空间增强

默认 `--windows-per-patient 4`：每轮每患者无放回抽取最多 4 个窗口，窗口不足时全部使用；每轮通过 seed + epoch 重新生成抽样。`0` 恢复全部窗口。多 GPU 使用相同全局抽样后分片，为保证各 rank 训练步数一致，会最少量重复补齐至 world size 的整数倍；补齐数在启动日志中可见。恢复训练时使用保存的下一 epoch，保持相同 seed 和采样参数即可复现对应轮次的样本选择。此参数是当前命令配置，不从旧 checkpoint 自动覆盖。

训练保留原有翻转，并以默认 0.5 概率对每视图独立应用小幅平面内仿射：旋转 ±10°，H/W 平移各不超过对应尺寸 5%，等比例缩放 0.9–1.1 倍。长轴同一窗口的全部帧与中心标签共享同一个变换；SAX 所有层与标签共享同一个 H/W 变换，不混合深度。图像双线性插值、标签最近邻，超出范围补 0。不同视图坐标系不同，不强制复用相同变换。未新增强度、gamma、噪声或偏置场增强。验证和测试关闭增强且覆盖全部窗口。

```bash
WINDOWS_PER_PATIENT=4 bash run.sh train-multi-gpu
WINDOWS_PER_PATIENT=8 bash run.sh train
# 对照：使用全部窗口，关闭新增仿射（保留原有翻转）
WINDOWS_PER_PATIENT=0 AFFINE_PROB=0 bash run.sh train
# 自定义较弱空间增强
ROTATION_DEGREES=5 TRANSLATION_FRACTION=0.03 SCALE_FRACTION=0.05 bash run.sh train
```

Python 参数对应 `--windows-per-patient`、`--affine-prob`、`--rotation-degrees`、`--translation-fraction`、`--scale-fraction`。无需重新预处理，模型结构未变化，仅当前通道时间窗、patch size=1 的 v14 checkpoint 可完整 resume；改变采样或增强后属于新的训练配置，建议另设 OUTPUT_DIR。resume 仍恢复旧学习率及调度器，不会自动重新升高学习率。

## 长轴空间适配层与 AdamW

2CH/4CH 主 decoder 每级为：共享上采样/卷积 → 视图独立 FiLM → 视图独立空间残差适配层。适配层使用 `3×3 depthwise conv → GroupNorm → SiLU → 1×1 pointwise conv`，输出与输入残差相加。三个尺度各有独立参数，两个视图也不共享，共 6 个适配层；pointwise 权重和 bias 零初始化，初始严格为恒等映射。适配后的特征用于后续上采样以及主分割头。SAX decoder 和辅助 decoder 不新增该模块。

训练使用 AdamW，默认 lr=1e-4、weight_decay=1e-4。只对卷积及线性权重（含 attention 投影）衰减；bias、归一化参数、位置/模态编码、解剖 token 与标量门控不衰减。可通过 `--weight-decay` 或 `WEIGHT_DECAY` 修改。完整 resume 恢复 checkpoint 中的 AdamW 状态和衰减值，命令行值不覆盖恢复状态。

```bash
OUTPUT_DIR=checkpoints_adapter WEIGHT_DECAY=1e-4 bash run.sh train-multi-gpu
RESUME=checkpoints_adapter/last_model_3d_multihead.pth OUTPUT_DIR=checkpoints_adapter EPOCHS=300 bash run.sh train-multi-gpu
```

无需重新预处理。结构标记为 `center_window_lax_joint_cross_patch1_deep_v14`；旧 v4 及更早 checkpoint 不能直接测试/完整 resume，也不会把旧 Adam 状态误载入 AdamW。请使用新的输出目录重新训练。

## 可选 FP16 混合精度

默认保持 FP32。训练和测试均可通过 `--fp16` 或 `FP16=1` 启用 CUDA FP16 自动混合精度；CPU 不支持此选项，会明确报错。模型参数和优化器状态保持 FP32，前向使用 autocast，分割/边界/辅助损失和指标使用 FP32；推理 softmax 与跨窗口概率平均也使用 FP32。

```bash
FP16=1 bash run.sh train
FP16=1 bash run.sh train-multi-gpu
FP16=1 bash run.sh test
FP16=1 RESUME=checkpoints/last_model_3d_multihead.pth EPOCHS=300 bash run.sh train-multi-gpu
```

训练使用 GradScaler，溢出时跳过优化器更新并调整缩放。checkpoint 保存并在 FP16 续训时恢复 scaler 状态。旧 FP32 同架构 checkpoint 仍能续训，首次启用 FP16 会提示重新初始化 scaler；FP16 checkpoint 也可用 FP32 继续训练或推理。精度由本次命令行决定，resume 时需再次传入 `--fp16` / `FP16=1`。本次不改变网络结构或缓存格式，无需重新预处理/从头训练。实际加速及显存收益取决于 GPU 和输入尺寸，不能保证提高 Dice。

## 统一损失入口与单尺度深度监督（v13）

所有训练损失位于 `utils/losses.py`：`dice_ce_loss_per_sample`、`boundary_loss_per_sample`、`masked_segmentation_loss`。`train.py` 只调用组合损失并执行优化；`utils/metrics.py` 只保留评估指标；模型只负责前向。

训练对每个有效样本—视图计算 `Dice + CE + 0.05 * boundary + 0.2 * auxiliary(Dice + CE) + 0.1 * d2(Dice + CE)`，再对有效样本—视图数量取平均。Dice 仍包含背景，CE 无类别权重；边界仅考虑有效前景类别，默认下采样 2、距离截断 10。缺失视图不参与监督。验证不计算辅助损失和深度监督损失。辅助 decoder 保留，与主 decoder 中间层深度监督是两回事。

主 decoder 在 d2（默认空间尺寸 64）上使用各视图独立的 1×1 分割头，长轴为 Conv2d，SAX 为 Conv3d。中间 logits 插值回标签尺寸，使用 Dice+CE，不额外添加 boundary loss。推理不运行中间头。未使用的 CE-only 模型损失函数及 DiceCE 包装类/函数保持删除。参数集合改变，新 checkpoint 标记为 `center_window_lax_joint_cross_patch1_deep_v14`；旧 checkpoint 不能直接测试或完整 resume，请使用新输出目录训练。预处理缓存可继续使用，FP16 和同版本完整 resume 继续支持。

## 通道时间窗与两阶段交互（v12）

长轴默认五帧作为输入通道，在首层卷积融合；不再逐帧编码或拼接时间 tokens。窗口长度 K 仍由预处理 YAML 决定。现有五帧窗口缓存无需重新预处理，主输出和辅助输出仍为中心帧分割。

两阶段 cross-attention 保留：先由 2CH/4CH 与对应解剖 tokens 双向交互，再由 SAX 与其相连解剖 tokens 双向交互。第二阶段不再次更新长轴 patches，但最终解剖 tokens 通过 FiLM 调制所有视图 decoder。模态掩码、空间适配层、FP16、统一损失入口均不变。

默认 token 数为 2CH 256、4CH 256、SAX 4096、解剖 5，总计 4613。新增长轴联合 self-attention 有 512²=262144 个矩阵位置；两阶段双向融合 cross-attention 矩阵位置合计 46080（包含被屏蔽位置，不计 fallback）；这不代表整网计算量，编码器 intra self-attention、卷积和 FFN 仍需计算。

空间 patch size 为 1：长轴使用 1×1、stride=1 卷积，SAX 使用 1×1×1、stride=1 卷积。每个瓶颈空间位置产生一个 token，无额外下采样或补零。交互后恢复原瓶颈网格并残差融合，分割输出尺寸不变。

首层卷积权重形状改变，且删除时间位置参数。新架构标记为 `center_window_lax_joint_cross_patch1_deep_v14`，旧 checkpoint 不能直接测试或完整 resume；请使用新输出目录训练。同版本支持完整 resume。例如：

```bash
FP16=1 OUTPUT_DIR=checkpoints_window_channels bash run.sh train-multi-gpu
```

## 训练时随机屏蔽 patch Key/Value

从头训练和 resume 均默认 `--token-drop-rate 0.05`。仅当样本有至少两个存在模态时，在每个存在视图内随机选择约 5% 的 patch，禁止解剖 tokens 读取这些 patch 的 Key/Value。每视图屏蔽数量四舍五入，至少保留一个 patch；极小网格可能不屏蔽。每次训练 forward 重新抽样。

全部 patch queries、CNN skips、中心帧标签、解剖 tokens 和原模态掩码保留。反向的 patch 读取解剖信息不受这个随机 mask 限制。验证/测试完全关闭随机屏蔽，无新增 loss，预处理不变。

```bash
TOKEN_DROP_RATE=0.05 FP16=1 OUTPUT_DIR=checkpoints_tokenmask bash run.sh train-multi-gpu
# 对照：关闭随机 token 屏蔽
TOKEN_DROP_RATE=0 FP16=1 OUTPUT_DIR=checkpoints_no_tokenmask bash run.sh train-multi-gpu
```

token 屏蔽选项本身不改变参数形状；当前联合长轴 attention 架构及单尺度深度监督版本为 v14。比例保存在 checkpoint 的 model_config；每次启动训练（包括 resume）均以本次参数为准，未指定时使用 0.05，不沿用 checkpoint 中的比例。显式设置 `TOKEN_DROP_RATE=0` 可关闭。resume 仍恢复原优化器和学习率调度器。


## 长轴 patches 直接交互（v12）

2CH 和 4CH patches 先拼接为 `[B,512,dim]`（默认每视图 256），通过新增的联合 self-attention 直接双向读取。缺失长轴视图的 keys 被屏蔽，SAX-only 样本跳过该层，缺失视图输出置零。随后保留原有两阶段：长轴与解剖 tokens 双向交互，再由 SAX 与其关联解剖 tokens 双向交互。

SAX 不加入长轴联合 self-attention，与长轴仍仅通过解剖 tokens 交换信息。原固定解剖连接保持；但 4CH patches 已融合 2CH 信息，因此心房 tokens 可经 4CH patches 间接接收 2CH 信息。最终解剖 tokens 继续通过 FiLM 调制所有视图 decoder。

随机 token 屏蔽仍只限制解剖 tokens 读取 patch Key/Value，不作用于新增长轴 self-attention。token 总数、五帧通道输入、patch=1、encoder/decoder 深度及 loss 均不变。新增 attention 参数使旧 checkpoint 无法完整 resume/直接测试，使用新输出目录重新训练：

```bash
FP16=1 OUTPUT_DIR=checkpoints_joint_lax bash run.sh train-multi-gpu
```


默认 `DEEP_WEIGHT=0.1`（Python 参数 `--deep-weight`）。直接运行 `bash run.sh train-multi-gpu` 即启用。对照实验保留相同的中间头，只把权重设为 0；固定其余配置、随机种子和训练预算：

```bash
DEEP_WEIGHT=0.1 OUTPUT_DIR=checkpoints_deep FP16=1 bash run.sh train-multi-gpu
DEEP_WEIGHT=0 OUTPUT_DIR=checkpoints_no_deep FP16=1 bash run.sh train-multi-gpu
```

深度监督权重记录在 checkpoint 的 train_args；resume 时以本次参数为准，未指定仍默认 0.1。新增参数使 v12 及更早 checkpoint 不兼容完整恢复，需从新目录训练；预处理缓存不变。


## Patch size=1（v14）

三个视图的 patch embedding 均为 kernel=1、stride=1，保留 CNN 三次下采样后的完整空间网格。默认 128 输入的 token 数为 256+256+4096+5=4613。五帧通道输入、长轴联合 attention、SAX 解剖交互、单尺度深度监督和随机 token 屏蔽保持不变，无教师学生蒸馏。

attention 计算与显存开销增加，尤其 SAX 模态内 attention。预处理缓存不变，但 patch 权重形状改变，旧 v13 及更早 checkpoint 不兼容直接测试或完整 resume，请重新训练：

```bash
FP16=1 OUTPUT_DIR=checkpoints_patch1 bash run.sh train-multi-gpu
```
