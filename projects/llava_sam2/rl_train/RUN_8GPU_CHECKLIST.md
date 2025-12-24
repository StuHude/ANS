# 八卡训练配置检查清单

在新服务器上运行八卡训练前，请检查并修改以下配置：

## 1. 必须修改的路径（在 run_rl_8gpu.sh 中）

### 1.1 模型路径
```bash
MODEL_PATH="/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
```
**修改为**: 你的服务器上Sa2VA模型的实际路径

### 1.2 数据路径
```bash
DATA_DIR="/data/xiaoyicheng/Sa2VA/data/GAR"
```
**修改为**: 你的服务器上GAR数据集的实际路径

### 1.3 Conda环境激活路径
```bash
source /home/xiaoyicheng/miniconda3/etc/profile.d/conda.sh
```
**修改为**: 你的服务器上miniconda3/anaconda3的实际路径
- 可能是 `/home/YOUR_USERNAME/miniconda3/...`
- 或者 `/opt/conda/...`
- 运行 `which conda` 可以找到conda路径

## 2. 可选修改项

### 2.1 端口号（如果29600被占用）
```bash
--master_port=29600
```
**修改为**: 其他未被占用的端口，如 29601, 29700 等

检查端口是否被占用：
```bash
netstat -tuln | grep 29600
```

### 2.2 GPU选择（如果不想用全部8卡）
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```
**修改为**: 你想用的GPU编号，例如：
- `0,1,2,3` (只用前4卡)
- `4,5,6,7` (只用后4卡)

同时修改：
```bash
--nproc_per_node=8
```
改为对应的GPU数量

## 3. 关键参数说明（不要改！）

### ⚠️ 这些参数已经优化过，不要修改：
```bash
NUM_GENERATIONS=2       # ✓ GRPO算法要求最少2个，否则会NaN
PER_GPU_BATCH_SIZE=1    # ✓ 针对48GB显存优化过
GRADIENT_ACCUM_STEPS=2  # ✓ 梯度累积，模拟更大batch
```

## 4. 启动命令

### 方法1: 前台运行（推荐用于测试）
```bash
bash projects/llava_sam2/rl_train/run_rl_8gpu.sh
```

### 方法2: 后台运行（推荐用于长时间训练）
```bash
nohup bash projects/llava_sam2/rl_train/run_rl_8gpu.sh > /tmp/sa2va_rl_8gpu.log 2>&1 &
```

### 方法3: 使用screen/tmux（推荐）
```bash
# 创建新session
screen -S sa2va_rl

# 运行训练
bash projects/llava_sam2/rl_train/run_rl_8gpu.sh

# 分离: Ctrl+A, 然后按D
# 重新连接: screen -r sa2va_rl
```

## 5. 监控命令

### 实时查看日志
```bash
tail -f /tmp/sa2va_rl_8gpu_full_epoch.log
```

### 查看GPU状态
```bash
watch -n 2 nvidia-smi
```

### 查看训练进度
```bash
grep "Epoch 1 Step" /tmp/sa2va_rl_8gpu_full_epoch.log | tail -20
```

### 检查是否有OOM错误
```bash
grep -i "out of memory\|OOM" /tmp/sa2va_rl_8gpu_full_epoch.log
```

## 6. 预期训练配置

- **总样本数**: 48,689
- **每epoch步数**: ~6,086 (48689 / 8)
- **有效batch size**: 16 (8 GPU × 1 batch × 2 accum)
- **Checkpoint保存**: 每1000步
- **显存使用**: ~45-48GB/GPU
- **训练时长**: 约6-12小时（取决于硬件）

## 7. 常见问题

### Q: 如何修改保存checkpoint的频率？
在train_sa2va_dual_loop.py中添加 `--save_steps` 参数：
```bash
--save_steps 500  # 每500步保存一次
```

### Q: 显存不足怎么办？
如果48GB显存的GPU出现OOM：
1. 检查是否有其他进程占用显存
2. 不要修改NUM_GENERATIONS（必须>=2）
3. 已经是最优配置，考虑用更大显存的GPU

### Q: 如何恢复训练？
脚本会自动每1000步保存checkpoint到输出目录，可以使用checkpoint恢复。
