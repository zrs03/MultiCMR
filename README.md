# MultiCMR

2CH/4CH 使用完整的 **2D+t** 序列，SAX 使用 3D 输入。三种模态通过原生 patch tokens、可学习 fusion tokens 和 5 个 anatomy tokens 融合，并保留缺失模态掩码。

| 模态 | 预处理数组 | 模型输入 | 分割 logits |
| --- | --- | --- | --- |
| 2CH | `[64,128,128] = [T,H,W]` | `[B,1,64,128,128]` | `[B,3,64,128,128]` |
| 4CH | `[64,128,128] = [T,H,W]` | `[B,1,64,128,128]` | `[B,6,64,128,128]` |
| SAX (`sa`) | `[128,128,128] = [D,H,W]` | `[B,1,128,128,128]` | `[B,4,128,128,128]` |

原始长轴 NIfTI 使用 `[H,W,T]`（也支持 `[H,W,1,T]`），先转换为 `[T,H,W]`，再重采样到 64 帧。原始 SAX 数组最后一轴移到最前；保留原有 SAX 堆叠语义，不另外推断或拆分心动周期。图像使用非零区域 z-score 和线性插值，标签使用最近邻插值。增强对整段长轴序列使用一致的空间翻转，不翻转时间顺序。

```bash
python -m pip install -r requirements.txt
bash run.sh preprocess
bash run.sh train
bash run.sh test
```

预处理默认写入 `preprocess/processed/{train,val,test}/*.npz` 和 `preprocess/matched.yaml`。每个 NPZ 保存 image、可选 label、原始 affine/shape、view 和布局信息。也可以只生成原始路径清单，由 Dataset 在线执行相同处理：

```bash
python -m preprocess.preprocess --manifest-only
```

训练、验证和推理均输入全部 64 帧，不随机抽帧。模型接收 `{'2ch': tensor, '4ch': tensor, 'sa': tensor}` 与 `[B,3]` 模态掩码。2CH/4CH 将 `[B,C,T,H,W]` 展开为 `[B*T,C,H,W]`，逐帧经过共享的 2D encoder、1×1 patch embedding、空间位置编码和模态内 Transformer。随后恢复时间维度，加入可学习的时间位置编码，按时间顺序拼接为 `[B,T*P2D,dim]`。两个长轴视图使用各自的 encoder 参数。

默认每帧的 patch 网格为 `16×16`，每个长轴序列保留 `64×16×16 = 16384` 个 tokens；SAX 的 3D 网格为 `16×16×16 = 4096` 个 tokens。拼接后进行跨帧、跨模态的联合 attention，包含 4096 个 fusion tokens 和 5 个 anatomy tokens，不要求三个模态 token 数量相同。`token_grid` 控制可学习位置网格的初始大小，不用于池化或削减原生 patches。

长序列 attention 按 query 分块计算，每个 query 仍关注全部可用 keys；训练时通过 checkpoint 重算 attention 分块来控制中间矩阵显存。这保留全局 attention 和所有 patch，但完整 2D+t 训练仍比单帧需要更多计算和显存。

解码时，融合后的长轴 tokens 还原成 `[B,dim,T,h,w]`，使用 2D decoder 逐帧解码，再恢复 `[B,classes,T,H,W]`。Fusion 特征沿 SAX 深度取均值后广播到各帧，避免把 SAX 深度当作长轴时间；各帧自身的融合 tokens 保留时序上下文。SAX 继续使用 3D encoder/decoder。解剖 token 注入、辅助分割头和深监督结构保留。

训练保持原有 7 种非空模态掩码随机采样，验证使用全模态。缺失模态不进入 encoder、不作为 attention key/value，也不计入损失和指标。Dice+CE、边界损失和指标对长轴逐帧计算后平均，边界距离只在帧内计算；SAX 使用 3D 计算。辅助/深监督默认权重为 `--aux-weight 0.2` 和 `--deep-weight 0.1`，边界损失可通过 `--boundary-weight 0` 关闭。

独立评估 `test.py` 每种模态组合执行一次完整序列 forward，直接输出长轴 `[64,H,W]` 与 SAX `[128,128,128]` 分割。支持 `full`、`random`、`both`、`all-seven` 模态组合；`--split test` 无标签也可导出预测。

```bash
python test.py --checkpoint checkpoints/best_model_3d_multihead.pth --split val --eval-mode all-seven
SPLIT=test bash run.sh test
python -m unittest discover -s tests -v
```

导出 NIfTI 使用模型坐标和单位 affine：长轴为 `[H,W,1,T]`，SAX 为 `[H,W,D]`，不直接代表原始物理坐标。`calculate_lv_ef.py` 要求恢复到原始 SAX 布局和 spacing 的分割，不能直接使用这些模型坐标导出文件。

尺寸选项改为 `--lax-size T H W` 和 `--sax-size D H W`，启动脚本对应 `LAX_SIZE`、`SAX_SIZE`。默认从 YAML/checkpoint 读取各模态尺寸；缓存尺寸有变化时需重新预处理。旧 `--input-size`、三路统一 3D encoder 的 checkpoint，以及不含时间位置编码的单帧 checkpoint 不兼容，需要用新结构重新训练。新 checkpoint 保存各模态尺寸、时间位置编码长度和完整模型配置。原先未生效的缺失模态合成/token 重建参数已移除。
