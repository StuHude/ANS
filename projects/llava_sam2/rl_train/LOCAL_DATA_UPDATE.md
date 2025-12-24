# 本地数据加载功能更新说明

## 更新内容

已经修改代码以支持从本地Arrow文件加载Grasp-Any-Region-Dataset，无需从HuggingFace在线下载。

## 修改的文件

### 1. `dataset.py` ✅
**新增功能**:
- 新增 `local_data_dir` 参数：指定本地数据目录
- 新增 `parts_to_load` 参数：指定要加载的Part文件夹
- 新增 `_load_from_local()` 方法：从本地Arrow文件加载
- 自动检测和合并多个Part

**向后兼容**: 原有的HuggingFace加载方式仍然可用

### 2. `train_sa2va_rl.py` ✅
**新增参数**:
- `--local_data_dir`: 本地数据目录路径
- `--parts_to_load`: 要加载的Part列表

### 3. `run_rl_train.sh` ✅
**新增配置**:
- `LOCAL_DATA_DIR`: 本地数据目录
- `PARTS_TO_LOAD`: Part列表（可选）
- 自动构建参数并传递给训练脚本

### 4. 新增文件

#### `LOCAL_DATA_GUIDE.md` ✅
详细的本地数据加载使用指南，包括：
- 数据结构要求
- 多种使用方法
- 常见问题解答
- 调试技巧

#### `test_dataset_loading.py` ✅
数据加载测试脚本，用于验证：
- 本地数据是否能正确加载
- 数据格式是否正确
- 样本能否正常读取

## 使用方法

### 快速测试

首先测试数据加载是否正常：

```bash
python projects/llava_sam2/rl_train/test_dataset_loading.py \
    --local_data_dir /data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287
```

预期输出：
```
============================================================
Testing Dataset Loading
============================================================

Loading from local directory: /data/xyc/.../9422475719852204c04762f299967c3a4ca58287
Parts to load: All available parts (auto-detect)
Loading from local directory: /data/xyc/.../9422475719852204c04762f299967c3a4ca58287
Parts to load: ['Fine-Grained-Dataset-Part1', ...]
Loaded 50000 samples from Fine-Grained-Dataset-Part1
...

✓ Successfully loaded dataset!
Total samples: 290000
```

### 开始训练

#### 方法1: 修改启动脚本（推荐）

编辑 `run_rl_train.sh`:
```bash
# 找到第20行左右
LOCAL_DATA_DIR="/data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287"
```

然后运行：
```bash
bash projects/llava_sam2/rl_train/run_rl_train.sh
```

#### 方法2: 直接使用命令行

```bash
torchrun --nproc_per_node=8 \
    projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --model_path /data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new \
    --local_data_dir /data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287 \
    --output_dir ./outputs/sa2va_grpo \
    --num_epochs 2
```

## 数据路径说明

### 当前检测到的数据位置

```
/data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/
└── snapshots/
    └── 9422475719852204c04762f299967c3a4ca58287/
        └── Fine-Grained-Dataset-Part1/
            └── data-00001-of-00044.arrow
```

### 你需要确认的事项

1. **确认所有Part的位置**

请运行以下命令找到所有Part：
```bash
find /data -type d -name "Fine-Grained-Dataset-Part*" 2>/dev/null
```

2. **如果Part在不同位置**

如果Part2-6在其他位置，你需要：
- 将它们移动到同一个父目录下，或
- 创建符号链接，或
- 分别加载每个Part（需要修改代码）

## 目录结构示例

### 理想结构（推荐）
```
/data/xyc/grasp_dataset/
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

使用：
```bash
--local_data_dir /data/xyc/grasp_dataset
```

### 当前结构（如果Part分散）

如果你的Part在不同位置，建议：

**选项A: 创建符号链接**
```bash
mkdir -p /data/xyc/grasp_dataset
cd /data/xyc/grasp_dataset
ln -s /path/to/Part1 Fine-Grained-Dataset-Part1
ln -s /path/to/Part2 Fine-Grained-Dataset-Part2
...
```

**选项B: 只使用已有的Part**
```bash
--local_data_dir /data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287
--parts_to_load Fine-Grained-Dataset-Part1  # 只加载Part1
```

## 性能对比

| 方式 | 首次加载 | 网络 | 优势 |
|-----|---------|------|------|
| HuggingFace在线 | 慢（下载） | 需要 | 自动管理 |
| 本地Arrow | 快 | 不需要 | 速度快，离线可用 |

## 验证步骤

1. **测试数据加载**
```bash
python projects/llava_sam2/rl_train/test_dataset_loading.py \
    --local_data_dir YOUR_PATH
```

2. **查看加载信息**
应该看到：
- 成功找到Part文件夹
- 加载的样本数量
- 样本格式正确（image, mask, caption）

3. **开始训练**
```bash
bash projects/llava_sam2/rl_train/run_rl_train.sh
```

## 常见问题

### Q: 只检测到Part1，其他Part在哪里？

**A**: 运行以下命令找到所有Part：
```bash
find /data -type d -name "Fine-Grained-Dataset-Part*" 2>/dev/null
```

如果其他Part确实不存在，你有两个选择：
1. 只使用Part1进行训练（数据量较小）
2. 下载完整的Part2-6

### Q: Part在不同的目录下怎么办？

**A**: 创建一个统一的目录并使用符号链接：
```bash
mkdir -p /data/xyc/grasp_dataset_unified
ln -s /path/to/Part1 /data/xyc/grasp_dataset_unified/Fine-Grained-Dataset-Part1
ln -s /path/to/Part2 /data/xyc/grasp_dataset_unified/Fine-Grained-Dataset-Part2
...
```

### Q: 如何只加载特定的Part？

**A**: 使用 `--parts_to_load` 参数：
```bash
--parts_to_load Fine-Grained-Dataset-Part1 Fine-Grained-Dataset-Part3 Fine-Grained-Dataset-Part5
```

### Q: 加载失败怎么办？

**A**:
1. 检查路径是否正确
2. 检查Arrow文件是否存在
3. 运行测试脚本查看详细错误
4. 查看 `LOCAL_DATA_GUIDE.md` 的调试部分

## 下一步

1. ✅ 测试数据加载
2. ✅ 确认Part位置
3. ✅ 修改配置文件
4. 🚀 开始训练！

## 文件清单

新增/修改的文件：
- ✅ `dataset.py` - 支持本地加载
- ✅ `train_sa2va_rl.py` - 新增参数
- ✅ `run_rl_train.sh` - 新增配置
- ✅ `README.md` - 更新说明
- ✅ `LOCAL_DATA_GUIDE.md` - 详细指南
- ✅ `test_dataset_loading.py` - 测试脚本
- ✅ `LOCAL_DATA_UPDATE.md` - 本文档

## 技术细节

### 数据加载流程
```python
1. 检查 local_data_dir
   ↓
2. 扫描 Fine-Grained-Dataset-Part* 文件夹
   ↓
3. 对每个Part使用 load_from_disk()
   ↓
4. 使用 concatenate_datasets() 合并
   ↓
5. 返回统一的Dataset对象
```

### Arrow文件格式
- Arrow是高效的列式存储格式
- 支持零拷贝读取
- 适合大规模数据集
- HuggingFace datasets库原生支持

## 总结

✅ **已完成**:
- 本地Arrow文件加载功能
- 多Part自动检测和合并
- 向后兼容HuggingFace加载
- 完整的测试和文档

🎯 **优势**:
- 不需要网络连接
- 加载速度更快
- 更灵活的数据管理
- 支持部分加载

📚 **资源**:
- 详细指南: `LOCAL_DATA_GUIDE.md`
- 测试脚本: `test_dataset_loading.py`
- 配置示例: `run_rl_train.sh`
