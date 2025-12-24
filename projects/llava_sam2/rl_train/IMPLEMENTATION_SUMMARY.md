# Sa2VA RL Training Implementation Summary

## 完成的工作

### 1. 代码审查

#### `_get_token_logprobs` 函数分析 ✅

**位置**: `/data/xiaoyicheng/Sa2VA/projects/llava_sam2/hf/models/modeling_sa2va_chat.py:584-623`

**功能**: 从HuggingFace `generate()` 的输出中提取每个token的log probability

**实现检查**:
```python
def _get_token_logprobs(self, generate_output):
    scores = generate_output.scores          # list of length T_new
    sequences = generate_output.sequences    # (B, T_total)

    # 处理空输出情况
    if scores is None or len(scores) == 0:
        return torch.zeros(...)

    # 对每个时间步
    for t, step_scores in enumerate(scores):
        step_logp = F.log_softmax(step_scores, dim=-1)  # (B, V)
        idx = T_total - T_new + t
        token_ids = sequences[:, idx]
        token_logp = step_logp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        logprobs.append(token_logp)

    return torch.stack(logprobs, dim=1)  # (B, T_new)
```

**✅ 实现正确**:
1. 正确处理了 `generate_output.scores` 和 `generate_output.sequences` 的对应关系
2. 使用 `log_softmax` 而不是 `softmax` 直接计算log概率，数值更稳定
3. 正确使用 `gather` 来获取实际生成token的log概率
4. 返回形状正确: (B, T_new)
5. 边界情况处理完善（空输出）

**建议**: 无需修改，可以直接用于RL训练。

### 2. 实现的模块

#### 2.1 数据加载器 (`dataset.py`)

**类**: `GraspAnyRegionDataset`
- 从HuggingFace加载 `HaochenWang/Grasp-Any-Region-Dataset`
- 支持自动缓存
- 返回格式: `{image: PIL.Image, mask: np.ndarray, caption: str}`

**函数**: `collate_fn_sa2va_rl`
- 批处理collation函数
- 保持数据为列表格式，在trainer中动态预处理

#### 2.2 奖励函数 (`reward_functions.py`)

**IOU奖励**:
- `compute_iou(mask1, mask2)`: 计算两个mask的交并比
- `iou_reward_batch()`: 批量计算IOU
- 范围: [0, 1]
- 用于Loop 1: mask → caption → mask'

**METEOR奖励**:
- `compute_meteor(reference, hypothesis)`: 计算caption相似度
- `meteor_reward_batch()`: 批量计算METEOR
- 使用NLTK实现
- 范围: [0, 1]
- 用于Loop 2: caption → mask → caption'

#### 2.3 EMA模型包装器 (`ema_model.py`)

**类**: `EMAModel`
- 实现指数移动平均更新: `ema_param = decay * ema_param + (1 - decay) * student_param`
- 默认decay=0.999
- 自动冻结EMA模型的梯度
- 支持模型属性代理访问

**更新方式**:
```python
ema_model.update(student_model)  # 在每次optimizer.step()后调用
```

#### 2.4 GRPO训练器 (`sa2va_grpo_trainer.py`)

**核心类**: `Sa2VAGRPOTrainer`

**双循环实现**:

**Loop 1: Mask → Caption → Mask' → IOU**
1. `rollout_mask_to_caption()`: 使用Model1从mask生成caption，获取logprobs
2. `compute_mask_to_caption_rewards()`: 使用Model2从caption生成mask'，计算IOU
3. `compute_grpo_loss()`: 计算GRPO损失

**Loop 2: Caption → Mask → Caption' → METEOR**
1. `rollout_caption_to_mask()`: 使用Model1从caption生成mask，获取logprobs
2. `compute_caption_to_mask_rewards()`: 使用Model2从mask生成caption'，计算METEOR
3. `compute_grpo_loss()`: 计算GRPO损失

**GRPO算法**:
- 每个样本生成G个rollouts (默认G=4)
- 组内reward归一化: `(rewards - mean) / std`
- PPO-style clipping: `min(ratio * adv, clip(ratio, 1±ε) * adv)`
- KL散度惩罚: `kl_coef * (ref_logprob - policy_logprob)`
- 总损失: `policy_loss + kl_penalty`

**可训练参数**:
- Projector (mlp1)
- LLM LoRA layers
- SAM2 mask decoder
- Text hidden fcs

**冻结参数**:
- Vision encoder
- LLM base (除LoRA外)
- SAM2 image encoder

#### 2.5 主训练脚本 (`train_sa2va_rl.py`)

**功能**:
- 命令行参数解析
- 分布式训练支持 (DDP)
- 模型加载和参数冻结
- 数据集加载
- Wandb日志记录
- 训练循环管理

**支持的训练模式**:
- 单GPU训练
- 多GPU DDP训练

### 3. 配置和文档

**配置类**: `Sa2VAGRPOConfig`
- 包含所有训练超参数
- 默认值已设置为合理值

**文档**:
- `README.md`: 详细使用说明
- `IMPLEMENTATION_SUMMARY.md`: 本文档
- 代码内注释完善

**启动脚本**: `run_rl_train.sh`
- 一键启动训练
- 自动检查依赖
- 配置参数清晰

## 使用方法

### 快速开始

```bash
# 方法1: 使用启动脚本（推荐）
bash projects/llava_sam2/rl_train/run_rl_train.sh

# 方法2: 直接运行Python脚本
torchrun --nproc_per_node=8 \
    projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --model_path /data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new \
    --dataset_name HaochenWang/Grasp-Any-Region-Dataset \
    --output_dir ./outputs/sa2va_grpo \
    --num_epochs 2 \
    --num_generations 4
```

### 关键参数调整

**内存优化**:
- 减少 `--num_generations` (从4降到2)
- 减少 `--per_device_batch_size`
- 启用gradient checkpointing (需要在模型配置中设置)

**训练稳定性**:
- 调整 `--kl_coef` (0.05-0.2)
- 调整 `--clip_range` (0.1-0.3)
- 调整 `--learning_rate` (1e-6到1e-4)

**任务平衡**:
- 调整 `--mask_to_caption_weight`
- 调整 `--caption_to_mask_weight`

## 训练流程

```
1. 加载预训练模型 (Sa2VA-4B)
   ↓
2. 冻结vision encoder和LLM base
   ↓
3. 初始化EMA模型
   ↓
4. 对于每个batch:
   ├─→ Loop 1: mask → caption (M1) → mask' (M2) → IOU reward
   ├─→ Loop 2: caption → mask (M1) → caption' (M2) → METEOR reward
   ├─→ 计算GRPO loss
   ├─→ Backprop + Optimizer step
   └─→ 更新EMA模型
   ↓
5. 保存checkpoint
```

## 预期输出

### 日志示例

```
loop1/loss: 0.234
loop1/policy_loss: 0.189
loop1/kl_div: 0.045
loop1/reward_mean: 0.567
loop1/reward_std: 0.123

loop2/loss: 0.198
loop2/policy_loss: 0.156
loop2/kl_div: 0.042
loop2/reward_mean: 0.623
loop2/reward_std: 0.098

total_loss: 0.432
learning_rate: 9.5e-6
```

### Checkpoint结构

```
outputs/sa2va_grpo/checkpoint_step_1000/
├── pytorch_model.bin        # Student model (Model 1)
├── ema_model.bin            # EMA model (Model 2)
└── training_state.bin       # Optimizer state
```

## 技术细节

### GRPO vs PPO

**GRPO优势**:
1. **无需值函数**: 直接使用组内奖励归一化作为优势估计
2. **更简单**: 不需要训练critic网络
3. **更稳定**: 组归一化减少方差

**实现细节**:
```python
# Normalize rewards within group
rewards_norm = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)

# Use as advantages
advantages = rewards_norm

# PPO-style clipping
ratio = exp(logprob - ref_logprob)
loss = -min(ratio * adv, clip(ratio, 1±ε) * adv)
```

### 为什么使用EMA

**优势**:
1. **稳定参考**: Model 2作为稳定的参考模型，减少训练波动
2. **自举学习**: Model 2的输出质量随训练逐渐提高
3. **无需额外存储**: 相比保存多个checkpoint，EMA只需一个额外模型

**更新策略**:
- 高decay (0.999): 平滑更新，稳定但响应慢
- 低decay (0.99): 快速更新，响应快但可能不稳定

## 潜在问题和解决方案

### 问题1: 奖励稀疏

**现象**: 大部分样本IOU/METEOR都接近0

**解决**:
- 添加shaped reward: `reward = α * iou + β * mask_confidence`
- 使用reward shaping: `new_reward = old_reward + λ * auxiliary_reward`

### 问题2: 模式崩塌

**现象**: 模型总是生成相同的caption/mask

**解决**:
- 增加KL系数 (`--kl_coef`)
- 添加熵正则化
- 增加temperature采样

### 问题3: 训练不稳定

**现象**: Loss剧烈波动或NaN

**解决**:
- 降低学习率
- 增加warmup steps
- 减小clip_range
- 检查gradient clipping

## 扩展方向

1. **多任务学习**: 添加更多任务循环 (如图像描述、VQA等)
2. **Curriculum Learning**: 从简单样本逐渐过渡到复杂样本
3. **Adaptive Weighting**: 动态调整任务权重
4. **Reward Engineering**: 设计更好的奖励函数

## 依赖检查

**必需依赖**:
```bash
torch>=2.0.0
transformers==4.42.3
datasets
wandb
nltk
tqdm
numpy
PIL
```

**安装命令**:
```bash
pip install datasets wandb nltk tqdm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 引用

如果这个实现对你有帮助，请引用：

```bibtex
@misc{sa2va_rl_2025,
  title={Sa2VA Reinforcement Learning Training with GRPO},
  author={Sa2VA Team},
  year={2025},
}
```

## 联系

如有问题或建议，请联系项目维护者或提issue。

---

**实现日期**: 2025-11-29
**实现者**: Claude Code Assistant
**版本**: 1.0
