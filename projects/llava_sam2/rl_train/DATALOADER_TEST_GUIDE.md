# Dataloader测试指南

## 已完成的工作

### ✅ 1. Dataloader逻辑验证

已使用Mock数据成功测试了dataloader的核心逻辑：

```bash
export PATH="/home/xiaoyicheng/miniconda3/bin:$PATH"
conda activate vlm
python projects/llava_sam2/rl_train/test_dataloader_mock.py
```

**测试结果**: ✅ 全部通过
- Dataset长度正确
- 单样本加载正常
- 批次collation正确
- PyTorch DataLoader兼容

### ✅ 2. 代码改进

已修复`dataset.py`以处理HuggingFace Dataset的数据格式：
- 支持PIL Image
- 支持numpy array
- 支持list（HF Dataset可能将numpy转为list）
- 自动转换mask为boolean类型

## 下载真实数据进行测试

### 方法1: 使用HuggingFace镜像下载样本

已准备好下载脚本`download_gar_simple.py`：

```bash
# 设置环境
export PATH="/home/xiaoyicheng/miniconda3/bin:$PATH"
conda activate vlm

# 下载前50个样本用于测试
python projects/llava_sam2/rl_train/download_gar_simple.py
```

**注意**:
- 使用HF镜像站 `https://hf-mirror.com`
- 只下载50个样本（约占完整数据集的很小部分）
- 数据保存在 `./data/gar_test_sample/`

### 方法2: 手动下载数据

如果自动下载失败，可以手动操作：

#### 步骤1: 安装huggingface-cli

```bash
pip install -U huggingface_hub
```

#### 步骤2: 设置镜像并下载

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 方式A: 下载整个数据集
huggingface-cli download --repo-type dataset HaochenWang/Grasp-Any-Region-Dataset \
    --local-dir ./data/gar_dataset \
    --local-dir-use-symlinks False

# 方式B: 只下载Fine-Grained-Dataset-Part1（推荐测试用）
huggingface-cli download --repo-type dataset HaochenWang/Grasp-Any-Region-Dataset \
    --include "Fine-Grained-Dataset-Part1/*" \
    --local-dir ./data/gar_dataset \
    --local-dir-use-symlinks False
```

#### 步骤3: 验证下载

```bash
ls -la ./data/gar_dataset/Fine-Grained-Dataset-Part1/
```

应该看到多个`.arrow`文件。

## 测试Dataloader

### 测试1: Mock数据（已通过✅）

```bash
python projects/llava_sam2/rl_train/test_dataloader_mock.py
```

### 测试2: 真实数据

下载完成后，运行：

```bash
python projects/llava_sam2/rl_train/test_dataset_loading.py \
    --local_data_dir ./data/gar_test_sample
```

或者如果使用手动下载的完整数据：

```bash
python projects/llava_sam2/rl_train/test_dataset_loading.py \
    --local_data_dir ./data/gar_dataset
```

### 预期输出

```
============================================================
Testing Dataset Loading
============================================================

Loading from local directory: ./data/gar_test_sample
Parts to load: All available parts (auto-detect)
Loading from local directory: ./data/gar_test_sample
Parts to load: ['Fine-Grained-Dataset-Part1']
Loaded 12345 samples from Fine-Grained-Dataset-Part1

✓ Successfully loaded dataset!
Total samples: 12345

============================================================
Testing sample loading...
============================================================

Sample 0:
  - Image: (512, 512) RGB
  - Mask: (512, 512) bool
  - Caption: A description of the masked region...

✓ All tests passed!
============================================================
```

## 数据集结构说明

### HuggingFace Dataset格式

GAR数据集在HuggingFace上的结构：

```
HaochenWang/Grasp-Any-Region-Dataset/
├── Fine-Grained-Dataset-Part1/
│   ├── data-00001-of-00044.arrow
│   ├── data-00002-of-00044.arrow
│   └── ... (共44个arrow文件)
├── Fine-Grained-Dataset-Part2/
│   └── ...
├── ... (Part3-6)
└── Relation-Dataset/  (不需要)
```

### 每个样本的字段

```python
{
    'image': PIL.Image,      # 图像
    'mask': numpy.ndarray,   # 二值mask (H, W)
    'caption': str,          # 描述文本
    # 可能还有其他字段...
}
```

## 常见问题

### Q1: 下载太慢或失败

**A**:
1. 确认HF_ENDPOINT环境变量设置正确: `https://hf-mirror.com`
2. 尝试手动下载
3. 或者只下载Part1进行测试

### Q2: Arrow文件在哪里？

**A**: 下载后，arrow文件会在：
```bash
./data/gar_test_sample/downloads/.../Fine-Grained-Dataset-Part1/
```

或者直接指向缓存目录（自动生成的hash路径）。

### Q3: 如何只加载部分数据？

**A**:
方式1 - 只下载部分Part：
```bash
--parts_to_load Fine-Grained-Dataset-Part1
```

方式2 - 使用split切片（在Python中）：
```python
dataset = load_dataset(
    "HaochenWang/Grasp-Any-Region-Dataset",
    split="train[:100]"  # 只加载前100个样本
)
```

## 当前下载状态

正在后台下载前50个样本用于测试...

检查下载进度：
```bash
# 查看缓存目录
ls -la ./data/gar_test_sample/

# 查看下载日志（如果有）
cat /tmp/gar_download.log
```

## 下一步

1. ✅ Mock数据测试通过
2. 🔄 正在下载真实数据样本
3. ⏳ 等待下载完成
4. ⏳ 使用真实数据测试dataloader
5. ⏳ 确认数据格式兼容性
6. 🚀 开始完整训练

## 快速测试命令总结

```bash
# 1. 激活环境
export PATH="/home/xiaoyicheng/miniconda3/bin:$PATH"
conda activate vlm

# 2. 测试dataloader逻辑（Mock数据）
python projects/llava_sam2/rl_train/test_dataloader_mock.py

# 3. 下载样本数据
python projects/llava_sam2/rl_train/download_gar_simple.py

# 4. 测试真实数据加载
python projects/llava_sam2/rl_train/test_dataset_loading.py \
    --local_data_dir ./data/gar_test_sample

# 5. 如果一切正常，开始训练！
bash projects/llava_sam2/rl_train/run_rl_train.sh
```

## 补充：手动检查数据

如果想手动查看arrow文件内容：

```python
from datasets import load_from_disk

# 加载arrow文件
dataset = load_from_disk("./data/gar_test_sample/...")

# 查看第一个样本
print(dataset[0])

# 查看数据结构
print(dataset.features)
print(len(dataset))
```

## 联系支持

如果遇到问题：
1. 查看本指南的"常见问题"部分
2. 查看 `LOCAL_DATA_GUIDE.md`
3. 检查下载日志
4. 确认网络和HF镜像设置
