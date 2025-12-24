# GAR数据处理Pipeline说明

## 架构设计

参考了`describe_anything_referring_dataset.py`的实现，我们采用**分离式设计**：

1. **原始数据加载** (`dataset_gar.py`)
2. **模型输入预处理** (`data_preprocessor.py`)

这种设计的优势：
- RL训练需要原始数据（计算reward，如IOU需要原始mask）
- 模型推理需要预处理数据
- 两个需求可以独立满足

## 组件详情

### 1. 原始数据加载器 (`dataset_gar.py`)

```python
from projects.llava_sam2.rl_train.dataset_gar import GraspAnyRegionDataset

dataset = GraspAnyRegionDataset(
    local_data_dir="/data/xiaoyicheng/Sa2VA/data/GAR",
    parts_to_load=None  # None = 自动加载所有Part
)

sample = dataset[0]
# Returns:
# {
#     'image': PIL.Image (RGB),
#     'mask': numpy.ndarray (H, W, bool),
#     'caption': str,
#     'category': str,
#     'image_id': str
# }
```

**功能：**
- ✅ 从Arrow文件直接加载
- ✅ RLE mask解码
- ✅ 从conversations提取caption
- ✅ 支持多Part自动拼接

### 2. 数据预处理器 (`data_preprocessor.py`)

```python
from projects.llava_sam2.rl_train.data_preprocessor import Sa2VADataPreprocessor

preprocessor = Sa2VADataPreprocessor()

# 准备mask->caption任务的输入
model_input = preprocessor.prepare_for_model(
    image=sample['image'],
    mask=sample['mask'],
    caption=sample['caption'],
    task="mask_to_caption"
)

# Returns:
# {
#     'pixel_values': torch.Tensor (1, 3, 448, 448),
#     'prompt_masks': torch.Tensor (1, 16, 16),
#     'vp_overall_mask': torch.Tensor (1,),
#     'prompt_text': str,
#     'region_pixels': [K],
#     'gt_caption': str
# }
```

**功能：**
- ✅ 图像resize到448×448并normalize（ImageNet mean/std）
- ✅ Mask聚合到16×16 token网格（使用adaptive_avg_pool2d）
- ✅ 构造特殊token格式：`<image> There are 1 part regions in the picture: region1<vp><IMG_CONTEXT>*K</vp>.\n{instruction}`
- ✅ 支持mask->caption和caption->mask两种任务

## 关键预处理步骤

### 图像预处理
```python
transforms.Compose([
    transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.229)),
])
```

### Mask预处理（最关键！）
```python
# 1. 转为torch tensor (1, H, W)
mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

# 2. Resize到图像尺寸 (1, 448, 448)
mask_resizer = transforms.Resize((448, 448), interpolation=InterpolationMode.NEAREST)
mask_tensor = mask_resizer(mask_tensor)

# 3. 聚合到token网格 (1, 16, 16)
pooled = F.adaptive_avg_pool2d(mask_tensor, (16, 16))

# 4. 二值化
prompt_masks = (pooled > 0.5).to(torch.uint8)

# 5. 计算region中的token数量K
region_pixels = [int(prompt_masks[0].sum().item())]
```

### 文本构造
```python
# Mask->Caption任务:
prompt = "<image> There are 1 part regions in the picture: region1<vp><IMG_CONTEXT>*181</vp>.\nPlease generate a detailed description for the given image region."

# Caption->Mask任务:
prompt = "<image> {caption}\nPlease segment the described region."
```

## 在RL训练中的使用

### Rollout阶段（生成）
```python
# 1. 从dataloader获取原始数据
sample = dataset[i]

# 2. 预处理为模型输入
model_input = preprocessor.prepare_for_model(
    image=sample['image'],
    mask=sample['mask'],
    task="mask_to_caption"
)

# 3. 模型推理（需要与Sa2VA模型集成）
# generated_caption = model.generate(
#     pixel_values=model_input['pixel_values'],
#     prompt_masks=model_input['prompt_masks'],
#     prompt_text=model_input['prompt_text']
# )
```

### Reward计算阶段
```python
# 使用原始数据计算reward
from projects.llava_sam2.rl_train.reward_functions import compute_iou, compute_meteor

# IOU reward (需要原始mask)
iou = compute_iou(sample['mask'], generated_mask)

# METEOR reward (需要原始caption)
meteor = compute_meteor(sample['caption'], generated_caption)
```

## 与describe_anything_referring_dataset的对比

| 项目 | describe_anything | 我们的实现 | 说明 |
|------|-------------------|-----------|------|
| 图像处理 | ✅ (448, 448) | ✅ (448, 448) | 相同 |
| Mask网格 | ✅ (16, 16) | ✅ (16, 16) | 相同 |
| 特殊token | ✅ | ✅ | 相同格式 |
| Tokenize | ✅ (video_lisa_encode_fn) | ❌ | 在RL训练脚本中处理 |
| Template | ✅ (template_map_fn) | ❌ | 在RL训练脚本中处理 |

**说明：**
- describe_anything返回完全tokenized的数据，直接可以送入模型训练
- 我们的实现返回**预处理但未tokenize**的数据，因为：
  1. RL训练需要原始数据计算reward
  2. Tokenization在RL训练脚本中更灵活

## 测试结果

```bash
# 测试原始数据加载
python test_gar_quick.py
# ✓ 3108 samples loaded
# ✓ Image, mask, caption all correct

# 测试预处理器
python test_preprocessor.py
# ✓ Image: (3, 448, 448)
# ✓ Mask: (1, 16, 16) with 181 tokens
# ✓ Prompt text formatted correctly
```

## 文件清单

```
projects/llava_sam2/rl_train/
├── dataset_gar.py              # 原始数据加载器
├── data_preprocessor.py        # 数据预处理器
├── test_gar_quick.py          # 测试原始数据加载
├── test_preprocessor.py       # 测试预处理器
├── reward_functions.py        # Reward函数（IOU, METEOR）
├── ema_model.py              # EMA模型
└── DATA_PIPELINE_SUMMARY.md  # 本文档
```

## 下一步

现在数据pipeline已完全准备好，可以进行：
1. ✅ 原始数据加载
2. ✅ Sa2VA格式预处理
3. 🔄 集成到RL训练脚本（需要tokenizer和model）
4. 🔄 实现LLM judge reward
5. 🔄 完整的RL训练循环
