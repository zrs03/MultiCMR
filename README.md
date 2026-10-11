# MultiCMR

2CH/4CH 使用以目标帧为中心的 **5 帧时间窗**，2D encoder/decoder 只预测中心帧；每个训练样本同时包含同病例 SAX 体积。原生 patch、fusion tokens、anatomy tokens 和缺失模态掩码保留，SAX 使用 3D encoder/decoder。

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

模型接收三视图字典和 `[B,3]` 模态掩码。长轴 `[B,1,K,H,W]` 转成 `[B,K,H,W]` 输入 2D encoder，亦支持直接传入后者。`temporal_size` 配置现在表示正奇数窗口长度，默认 5。两个长轴视图仍各自使用 encoder、共用 2D decoder、各自使用中心帧分类头；SAX 使用独立 3D 分支。辅助头、深监督头也只监督长轴中心帧。

默认尺寸下，每个长轴窗口生成 256 个 patch tokens，SAX 为 4096 个；1×1 patch embedding、空间位置编码、模态内 Transformer、联合 attention、4096 个 fusion tokens 和 5 个 anatomy tokens 保留。时间相对位置由窗口通道顺序表达。共享三维 fusion 特征沿深度平均后投影到长轴二维网格。attention 使用 query 分块和训练 checkpoint 重算。

训练仍均匀采样 7 种非空模态掩码；每条数据包含 SAX，但随机掩码可将它屏蔽。验证使用完整模态。缺失视图不编码、不作为 attention key/value、不计入监督。长轴 Dice+CE、边界损失及指标为中心帧二维计算；SAX 为三维计算。窗口内所有帧与中心标签采用一致的空间翻转。

日志中的训练/验证数量现在是**窗口数**。一个 epoch 遍历所有窗口，步骤数比原来的病例级 epoch 大；SAX 在每个窗口参与监督。训练脚本的 val Dice 是窗口加权统计（SAX 也按窗口计数），与独立 test.py 的病例级重建分数不同，帧数不同的病例权重不同。原来的 200 epochs 不再代表相同训练预算，请结合窗口总数调整 EPOCHS。

## 完整序列评估

```bash
bash run.sh test
SPLIT=test bash run.sh test
EVAL_MODE=all-seven bash run.sh test
python -m unittest discover -s tests -v
```

test.py 读取病例级完整序列，构造与预处理一致的成对时间窗，每次加同一份 SAX。一个病例在同一评估场景的所有窗口固定使用相同掩码。长轴中心帧预测按原始时间顺序重建，重复中心帧先平均类别概率再 argmax；SAX 对所有窗口的类别概率取平均后 argmax，每病例只评估一次。SAX-only 场景只执行一次网络。为控制 SAX 和 attention 显存，当前推理按窗口顺序执行，完整病例比原先一次全序列 forward 更慢。

支持 full、random、both、all-seven；random 场景各视图病例子集可能不同。导出仍为模型空间 NIfTI：长轴 `[H,W,1,T_native]`，SAX `[H,W,D]`，使用单位 affine。HD95 为模型网格单位，并非毫米。`calculate_lv_ef.py` 要求原始 SAX 布局和 spacing，不能直接使用这些导出文件。

尺寸选项为 `--lax-size K H W`、`--sax-size D H W`，启动脚本对应 `LAX_SIZE`、`SAX_SIZE`。train/test 默认从 YAML/checkpoint 读取配置，改变窗口或空间尺寸须重新预处理。新 checkpoint 标记为 `center_window_film_lax_3d_sax_v2`。旧 64 通道整序列输出、逐帧时间 token、三路 3D 等 checkpoint 均不能用于此结构，需重新训练。

## 断点续训

每轮结束后保存 `last_model_3d_multihead.pth`，验证 Dice 改善时另存 `best_model_3d_multihead.pth`。两者均包含模型、Adam、学习率调度器、已完成 epoch、历史最佳 Dice 及各 rank 的随机数状态。使用临时文件加原子替换保存，避免写入中断破坏上一次 checkpoint。

```bash
python train.py --resume checkpoints/last_model_3d_multihead.pth --epochs 300
RESUME=checkpoints/last_model_3d_multihead.pth EPOCHS=300 bash run.sh train
RESUME=checkpoints/last_model_3d_multihead.pth EPOCHS=300 bash run.sh train-multi-gpu
```

`--epochs` / `EPOCHS` 是目标总轮数：checkpoint 完成第 100 轮时，以上命令执行第 101–300 轮。日志追加而非清空。模型结构自动采用 checkpoint 配置，数据窗口尺寸、类别和视图顺序必须匹配；优化器恢复保存的学习率，`--lr` 不会覆盖完整续训的学习率。数据路径、batch size、worker 数、损失权重等沿用当前命令，建议与原训练保持一致。

支持从 best 或 last 续训；last 通常包含最近进度。同架构 FiLM checkpoint 只有模型权重时会明确告警：加载权重，但优化器/调度器重新初始化，epoch 从 1 开始。旧 cross-attention 和 64 通道架构仍不兼容。更改 GPU 数量可继续训练，但随机数流无法按原 rank 对应恢复，会告警；不保证与原训练轨迹一致。续训从完整 epoch 边界开始，不恢复中断 epoch 内尚未保存的 batch。

指定新的 OUTPUT_DIR 时，新 best 文件只有超过历史最佳分数才会写入；历史最佳文件仍位于原目录。相同数据、参数、worker 和 world size 是复现连续训练轨迹的前提。

## 解剖 token FiLM 调制

联合 Transformer 仍更新 5 个解剖 tokens。主 decoder 的三个尺度使用 FiLM 替代空间 query 对解剖 tokens 的 cross-attention：`F_out = (1 + delta_gamma) * F + beta`。2CH 选取心肌/左室腔 2 个 token，4CH 使用全部 5 个，SAX 使用心肌/左室腔/右室腔 3 个。按固定解剖顺序拼接选定 token，经 LayerNorm 和 MLP 生成每通道的缩放与偏移，广播到所有空间位置。每视图、每尺度都有独立生成器；2CH/4CH 继续共享解码卷积。

FiLM 最后一层权重和偏置零初始化，因此初始为恒等映射。主分割和深监督经调制特征输出；辅助 decoder 仍不使用解剖调制。模态掩码、patch 与 fusion token 机制保留。这是通道级全局条件调制，不保证某个 token 仅影响对应器官像素。

窗口预处理缓存可继续使用，无须重新拆分。旧 cross-attention checkpoint 不支持直接测试或完整 resume，需重新训练 FiLM 模型；建议指定新的 OUTPUT_DIR 避免覆盖旧实验。
