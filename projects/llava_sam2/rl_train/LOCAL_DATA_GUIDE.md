# 使用本地数据集训练指南

## 背景

你已经下载了Grasp-Any-Region-Dataset的Fine-Grained-Dataset-Part1到Part6的Arrow文件，现在需要从本地加载这些数据。

## 当前数据位置

根据系统检测，当前已下载的数据：
```
/data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287/
└── Fine-Grained-Dataset-Part1/
    └── data-00001-of-00044.arrow
```

**注意**: 目前只检测到Part1，请确认其他Part（Part2到Part6）的位置。

## 数据结构要求

你的数据目录应该是这样的结构：

```
/path/to/your/grasp_dataset/
├── Fine-Grained-Dataset-Part1/
│   └── data-00001-of-00044.arrow
├── Fine-Grained-Dataset-Part2/
│   └── data-00001-of-00044.arrow
├── Fine-Grained-Dataset-Part3/
│   └── data-00001-of-00044.arrow
├── Fine-Grained-Dataset-Part4/
│   └── data-00001-of-00044.arrow
├── Fine-Grained-Dataset-Part5/
│   └── data-00001-of-00044.arrow
└── Fine-Grained-Dataset-Part6/
    └── data-00001-of-00044.arrow
```

## 使用方法

### 方法1: 修改启动脚本 (推荐)

1. 编辑 `projects/llava_sam2/rl_train/run_rl_train.sh`

2. 找到这一行（第20行左右）：
```bash
LOCAL_DATA_DIR="/path/to/your/grasp_dataset"  # CHANGE THIS to your actual path
```

3. 修改为你的实际路径：
```bash
LOCAL_DATA_DIR="/data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287"
```

4. （可选）如果你只想加载特定的Part，取消注释并修改：
```bash
PARTS_TO_LOAD="Fine-Grained-Dataset-Part1 Fine-Grained-Dataset-Part2"
```

5. 运行训练：
```bash
bash projects/llava_sam2/rl_train/run_rl_train.sh
```

### 方法2: 命令行参数

直接在命令行指定参数：

```bash
# 单GPU训练
python projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --model_path /data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new \
    --local_data_dir /data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287 \
    --output_dir ./outputs/sa2va_grpo \
    --num_epochs 2 \
    --num_generations 4

# 多GPU训练
torchrun --nproc_per_node=8 \
    projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --model_path /data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new \
    --local_data_dir /data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287 \
    --output_dir ./outputs/sa2va_grpo \
    --num_epochs 2 \
    --num_generations 4
```

### 方法3: 只加载特定Part

如果你只想加载Part1-3：

```bash
python projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --local_data_dir /data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287 \
    --parts_to_load Fine-Grained-Dataset-Part1 Fine-Grained-Dataset-Part2 Fine-Grained-Dataset-Part3 \
    --model_path /data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new \
    --output_dir ./outputs/sa2va_grpo
```

## 自动检测机制

如果你不指定 `--parts_to_load`，代码会：
1. 自动扫描 `LOCAL_DATA_DIR` 目录
2. 找到所有 `Fine-Grained-Dataset-Part*` 文件夹
3. 按字母顺序加载
4. 自动合并所有Part的数据

## 数据加载流程

```
1. 检查 LOCAL_DATA_DIR 是否存在
   ↓
2. 扫描或使用指定的Part文件夹
   ↓
3. 对每个Part：
   ├─→ 尝试直接加载 (load_from_disk)
   └─→ 如果失败，尝试加载train子目录
   ↓
4. 合并所有Part的数据 (concatenate_datasets)
   ↓
5. 返回统一的Dataset对象
```

## 预期输出示例

当正确配置后，训练开始时你应该看到：

```
Loading from local directory: /data/xyc/.../9422475719852204c04762f299967c3a4ca58287
Parts to load: ['Fine-Grained-Dataset-Part1', 'Fine-Grained-Dataset-Part2', ...]
Loaded 50000 samples from Fine-Grained-Dataset-Part1
Loaded 48000 samples from Fine-Grained-Dataset-Part2
...
Concatenating 6 parts...
Loaded 290000 samples
```

## 数据格式

每个样本包含：
- `image`: PIL.Image对象
- `mask`: numpy array (H, W) 布尔类型
- `caption`: 字符串

## 常见问题

### Q1: 找不到Fine-Grained-Dataset-Part文件夹

**A**: 检查路径是否正确，确保路径指向包含Part文件夹的**父目录**。

```bash
# 正确 ✓
LOCAL_DATA_DIR="/path/to/grasp_dataset"  # 这个目录下有Part1, Part2...

# 错误 ✗
LOCAL_DATA_DIR="/path/to/grasp_dataset/Fine-Grained-Dataset-Part1"  # 太深了
```

### Q2: Part文件夹找到了但加载失败

**A**: 检查Part文件夹内部结构。可能有两种情况：
1. Arrow文件直接在Part文件夹下
2. Arrow文件在Part文件夹的`train`子目录下

代码会自动尝试这两种情况。

### Q3: 只有Part1被检测到

**A**:
1. 检查是否真的下载了Part2-6
2. 使用 `ls` 命令确认：
```bash
ls /data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287/
```

### Q4: 想要手动指定Part的顺序

**A**: 使用 `--parts_to_load` 参数，按你想要的顺序列出：
```bash
--parts_to_load Fine-Grained-Dataset-Part3 Fine-Grained-Dataset-Part1 Fine-Grained-Dataset-Part5
```

### Q5: 数据太大，只想用一部分训练

**A**: 使用 `--parts_to_load` 只加载部分Part：
```bash
--parts_to_load Fine-Grained-Dataset-Part1 Fine-Grained-Dataset-Part2
```

## 调试技巧

如果遇到问题，可以添加Python打印来调试：

```python
# 在train_sa2va_rl.py中临时添加
print(f"LOCAL_DATA_DIR: {args.local_data_dir}")
print(f"PARTS_TO_LOAD: {args.parts_to_load}")

# 或者检查目录内容
import os
print(os.listdir(args.local_data_dir))
```

## 从HuggingFace切换到本地

如果你之前使用HuggingFace在线加载，切换到本地非常简单：

**之前（在线加载）**:
```bash
python train_sa2va_rl.py \
    --dataset_name HaochenWang/Grasp-Any-Region-Dataset
```

**现在（本地加载）**:
```bash
python train_sa2va_rl.py \
    --local_data_dir /path/to/your/data \
    # dataset_name参数会被忽略
```

## 性能对比

| 加载方式 | 首次加载时间 | 后续加载时间 | 网络需求 |
|---------|------------|------------|---------|
| HuggingFace | 很慢（下载） | 快（缓存） | 需要 |
| 本地Arrow | 快 | 快 | 不需要 |

**推荐**: 使用本地Arrow文件，避免网络问题和下载时间。

## 完整示例命令

基于你当前的环境，完整的训练命令应该是：

```bash
torchrun --nproc_per_node=8 \
    projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --model_path /data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new \
    --local_data_dir /data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287 \
    --output_dir ./outputs/sa2va_grpo_$(date +%Y%m%d_%H%M%S) \
    --num_epochs 2 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_generations 4 \
    --wandb_project sa2va-rl
```

或者修改 `run_rl_train.sh` 后直接运行：
```bash
bash projects/llava_sam2/rl_train/run_rl_train.sh
```
