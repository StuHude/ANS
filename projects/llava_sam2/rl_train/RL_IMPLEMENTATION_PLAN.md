# Sa2VA RL训练实现方案

## 当前状况分析

### 1. 数据集
- ✅ GAR数据集已部分下载到 `/data/xiaoyicheng/Sa2VA/data/GAR/`
- ✅ Fine-Grained-Dataset-Part1: 3个arrow文件（约1.5GB）
- ✅ Fine-Grained-Dataset-Part2: 1个arrow文件
- ✅ Dataloader逻辑已实现并通过Mock数据测试

### 2. R1-V框架分析
- R1-V位于 `/data/xiaoyicheng/Sa2VA/R1-V/`
- 使用TRL库的`GRPOTrainer`和`GRPOConfig`
- 专为`Qwen2VLForConditionalGeneration`设计
- 有自定义trainer: `Qwen2VLGRPOTrainer`
- 支持单模型GRPO训练

### 3. 关键差异

**R1-V框架 vs 您的需求：**

| 项目 | R1-V | 您的需求 |
|------|------|----------|
| 模型 | Qwen2VL | Sa2VA-4B |
| 模型数量 | 单模型 | 双模型(EMA关系) |
| 训练循环 | 单任务 | 双循环(mask↔caption) |
| Reward | 单一reward函数 | 多重reward(IOU+METEOR+LLM judge) |
| 框架 | TRL GRPO | 需定制 |

## 实现方案

### 方案A: 使用R1-V框架（推荐，但需要大量修改）

**优点：**
- 复用成熟的GRPO实现
- 与vlm环境兼容

**需要修改：**

1. **创建Sa2VAGRPOTrainer**（参考`Qwen2VLGRPOTrainer`）
   - 适配Sa2VA模型的forward
   - 实现双模型管理(policy model + EMA model)
   - 处理mask和caption两种输入输出

2. **实现自定义reward函数**
   - IOU reward（mask对比）
   - Caption reward = 0.25 * METEOR + 0.75 * LLM_judge
   - 集成OpenAI API作为judge

3. **实现双循环训练**
   - Loop 1: mask → caption → mask' → IOU reward
   - Loop 2: caption → mask → caption' → caption reward
   - 需要在trainer中自定义step逻辑

### 方案B: 自定义实现（已部分完成）

**优点：**
- 完全控制训练流程
- 针对Sa2VA优化

**当前状态：**
- ✅ Dataset加载器
- ✅ EMA模型实现
- ✅ 基础reward函数（IOU, METEOR）
- ❌ LLM judge reward（需要实现）
- ❌ GRPO loss计算（已实现但需要验证）
- ❌ 与R1-V集成

## LLM Judge实现（参考describe-anything）

### describe-anything的评测方法

```python
# 来自 /data/xiaoyicheng/repos/describe-anything/evaluation/eval_model_outputs.py

def evaluate_caption_similarity(pred_caption, gt_caption, llm_client):
    """
    使用LLM judge评估两个caption的相似度

    返回: score (0-1之间的浮点数)
    """
    # 1. 基于gt_caption生成问答对
    # 2. 使用pred_caption回答问题
    # 3. 计算正确率作为相似度分数

    # 核心公式（来自describe-anything):
    score = (sum(positive_scores) + sum(negative_scores)) / (num_positives + num_negatives)

    return score
```

### 集成到Reward计算

```python
def compute_caption_reward(generated_caption, gt_caption):
    """
    Reward = 0.25 * METEOR + 0.75 * LLM_judge_score
    """
    # METEOR score
    meteor = compute_meteor(generated_caption, gt_caption)

    # LLM judge score
    llm_score = evaluate_caption_similarity(
        generated_caption,
        gt_caption,
        llm_client=openai_client
    )

    reward = 0.25 * meteor + 0.75 * llm_score
    return reward
```

## 数据集问题

### 当前问题
您下载的数据集是 `nvidia/describe-anything-dataset`，但代码中使用的是 `HaochenWang/Grasp-Any-Region-Dataset`

**这是两个不同的数据集！**

- `nvidia/describe-anything-dataset`: NVIDIA的描述数据集
- `HaochenWang/Grasp-Any-Region-Dataset`: GAR分割数据集

### 解决方案

**选项1**: 修改dataloader适配nvidia/describe-anything-dataset
- 需要了解该数据集的格式
- 可能需要不同的预处理

**选项2**: 下载正确的GAR数据集
- 继续下载 `HaochenWang/Grasp-Any-Region-Dataset`

## 建议的实施步骤

### 第一阶段：验证数据

1. **确认数据集**
   ```bash
   # 检查nvidia/describe-anything-dataset的格式
   python -c "from datasets import load_from_disk; \
              ds = load_from_disk('/data/xiaoyicheng/Sa2VA/data/GAR/Fine-Grained-Dataset-Part1'); \
              print(ds.features); print(ds[0].keys())"
   ```

2. **适配dataloader**
   - 根据实际数据格式修改dataset.py

### 第二阶段：实现LLM Judge

1. 参考 `/data/xiaoyicheng/repos/describe-anything/evaluation/eval_model_outputs.py`
2. 创建简化版的caption评估函数
3. 集成OpenAI API

### 第三阶段：选择训练框架

**如果使用R1-V:**
- 创建`Sa2VAGRPOTrainer`
- 需要2-3天开发时间

**如果使用自定义:**
- 完善现有代码
- 添加LLM judge
- 需要1-2天开发时间

### 第四阶段：测试运行

1. 小批量数据测试（10-100样本）
2. 验证reward计算正确
3. 验证EMA更新正确
4. 完整训练

## 关键代码位置

### 当前项目
- Dataloader: `projects/llava_sam2/rl_train/dataset.py`
- Rewards: `projects/llava_sam2/rl_train/reward_functions.py`
- EMA: `projects/llava_sam2/rl_train/ema_model.py`
- Trainer: `projects/llava_sam2/rl_train/sa2va_grpo_trainer.py`

### R1-V框架
- GRPO实现: `/data/xiaoyicheng/Sa2VA/R1-V/src/r1-v/src/open_r1/grpo.py`
- Trainer: `/data/xiaoyicheng/Sa2VA/R1-V/src/r1-v/src/open_r1/trainer/`

### describe-anything
- LLM评测: `/data/xiaoyicheng/repos/describe-anything/evaluation/eval_model_outputs.py`

## 下一步行动

请您确认：

1. **数据集选择**
   - 使用nvidia/describe-anything-dataset（当前已下载）？
   - 还是下载HaochenWang/Grasp-Any-Region-Dataset？

2. **框架选择**
   - 使用R1-V框架（需要大量适配）？
   - 还是使用自定义实现（快速但需要自己实现GRPO）？

3. **LLM Judge API**
   - 使用OpenAI API？
   - 还是本地部署的LLM？
   - API endpoint和key如何配置？

确认后我可以继续实现相应的代码。
