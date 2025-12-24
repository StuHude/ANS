import argparse
import copy
import math
import os
import sys
from pathlib import Path
import json

import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
import random
from safetensors.torch import load_file

# 把项目根目录加到 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from transformers import AutoTokenizer

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset

from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel


def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='RL hf checkpoint path (e.g. work_dirs/.../checkpoint-1000).')
    parser.add_argument(
        '--dataset',
        choices=['refcoco', 'refcoco_plus', 'refcocog'],
        default='refcoco',
        help='Specify a ref dataset')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco_plus'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
}

IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'
DATA_PATH = './data/ref_seg/'


def to_numpy_mask(x) -> np.ndarray:
    """兼容 torch.Tensor / numpy.ndarray，统一成 uint8 的 HxW numpy 掩码（阈值0.5）"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError(f"Unknown mask type: {type(x)}")
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expect 2D mask after squeeze, got shape {x.shape}")
    if x.dtype != np.uint8:
        x = (x > 0.5).astype(np.uint8)
    return x


def to_list_of_masks(arr) -> list:
    """把输入统一为 [H×W, H×W, ...] 的列表（numpy.uint8）"""
    if arr is None:
        return []
    if isinstance(arr, (list, tuple)):
        return [to_numpy_mask(m) for m in arr]
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)
    if arr.ndim == 2:
        return [to_numpy_mask(arr)]
    elif arr.ndim == 3:
        return [to_numpy_mask(arr[i]) for i in range(arr.shape[0])]
    else:
        raise ValueError(f"Unsupported mask shape: {arr.shape}")


def mask_to_rle(mask_list):
    """mask_list: [H×W, H×W, ...]；返回 COCO RLE 列表"""
    rle = []
    for m in mask_list:
        m = np.asfortranarray(m.astype(np.uint8))
        enc = _mask.encode(m)  # 对 2D 数组返回 dict
        enc['counts'] = enc['counts'].decode()
        rle.append(enc)
    return rle


def load_rl_state_dict_safetensors(ckpt_dir: str):
    """
    从 HF 分片 safetensors checkpoint 目录里，把所有 shard 合成一个 state_dict。

    ckpt_dir/
        model-00001-of-00002.safetensors
        model-00002-of-00002.safetensors
        model.safetensors.index.json
    """
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Cannot find index file: {index_path}")

    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]  # param_name -> shard_filename
    shard_files = sorted(set(weight_map.values()))

    merged_state_dict = {}
    for shard in shard_files:
        shard_path = os.path.join(ckpt_dir, shard)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
        print(f"  Loading shard: {shard_path}")
        shard_state = load_file(shard_path, device="cpu")
        merged_state_dict.update(shard_state)

    print(f"  Total params loaded from RL checkpoint: {len(merged_state_dict)}")
    return merged_state_dict


def build_model_and_tokenizer(rl_ckpt_path: str, rank: int = 0):
    """
    1) 从 base Sa2VA 模型构建结构
    2) wrap_llm_lora(r=128, lora_alpha=256, lora_dropout=0.05)
    3) 把 RL checkpoint 权重（safetensors）load 进来
    4) 准备 tokenizer & generation config
    """
    BASE_HF_MODEL_PATH = (
        "/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/"
        "Nemotrontiaozheng/Sa2VA-main/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
    )

    print(f"[Rank {rank}] Loading base Sa2VA model from {BASE_HF_MODEL_PATH}")
    model = Sa2VAChatModel.from_pretrained(
        BASE_HF_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,   # ★ 绝对不要用 True，避免 meta tensor
        use_flash_attn=True,
        trust_remote_code=True,
    )

    # 与训练代码对齐：wrap_llm_lora
    print(f"[Rank {rank}] Wrapping language model with LoRA (same as RL training)")
    model.wrap_llm_lora(r=128, lora_alpha=256, lora_dropout=0.05)

    # 载入 RL checkpoint 权重
    print(f"[Rank {rank}] Loading RL checkpoint state_dict from {rl_ckpt_path}")
    rl_state = load_rl_state_dict_safetensors(rl_ckpt_path)

    missing, unexpected = model.load_state_dict(rl_state, strict=False)

    print(f"[Rank {rank}] load_state_dict finished.")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("  e.g. missing[0:10]:", missing[:10])
    if len(unexpected) > 0:
        print("  e.g. unexpected[0:10]:", unexpected[:10])

    # tokenizer 用 base 模型那套（跟训练一致）
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_HF_MODEL_PATH,
        trust_remote_code=True,
    )

    # 与训练对齐：preparing_for_generation
    MAX_COMPLETION_LENGTH = 2048
    model.preparing_for_generation(tokenizer, max_new_tokens=MAX_COMPLETION_LENGTH)
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.max_new_tokens = MAX_COMPLETION_LENGTH
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'generation_config'):
        if model.language_model.generation_config is not None:
            model.language_model.generation_config.max_new_tokens = MAX_COMPLETION_LENGTH

    print(f"[Rank {rank}] Model prepared for generation (max_new_tokens={MAX_COMPLETION_LENGTH})")

    return model, tokenizer


def main():
    args = parse_args()

    # ====== 分布式初始化（和你原来一样）======
    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # ====== 构建模型 + 加载 RL 权重 ======
    rl_ckpt_path = args.model_path
    model, tokenizer = build_model_and_tokenizer(rl_ckpt_path, rank=rank)
    model = model.eval().cuda()

    # ====== 数据集 ======
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]

    dataset = RESDataset(
        image_folder=IMAGE_FOLDER,
        dataset_name=dataset_info['dataset_name'],
        data_path=DATA_PATH,
        split=args.split,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(
        per_rank_samples * rank,
        min(n_samples, per_rank_samples * (rank + 1))
    )

    debug_printed = False  # 只在 rank==0 的第一条样本上打印一次

    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]

        # ---- GT 处理：统一成 RLE 列表 ----
        gt_masks_list = to_list_of_masks(data_batch['gt_masks'])
        gt_rle = mask_to_rle(gt_masks_list)

        prediction = {
            'img_id': data_batch['img_id'],
            'gt_masks': gt_rle
        }

        texts = data_batch['text']
        # 清理出给模型的 batch
        del data_batch['img_id'], data_batch['gt_masks'], data_batch['text']

        pred_masks = []
        for ti, text in enumerate(texts):
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text

            # 这里直接用 Sa2VAChatModel 的 predict_forward
            out = model.predict_forward(**_data_batch, tokenizer=tokenizer)
            pred_mask = out.get('prediction_masks', None)

            # ---- 自检：只打印一次（rank==0 & 第一条样本的第一个文本）----
            if (not debug_printed) and rank == 0:
                print("[DEBUG] prediction_masks type:", type(pred_mask))
                if isinstance(pred_mask, (list, tuple)) and len(pred_mask) > 0:
                    print("[DEBUG] prediction_masks[0] type:", type(pred_mask[0]),
                          "shape:", getattr(pred_mask[0], "shape", None))
                else:
                    print("[DEBUG] prediction_masks empty or not list/tuple.")
                debug_printed = True

            if pred_mask is None or len(pred_mask) == 0:
                print("No seg pred !!!")
                pred_masks.append(None)
            else:
                # 只用第一张；若要融合多张可以之后自己改
                first_mask = to_numpy_mask(pred_mask[0])
                rle_list = mask_to_rle([first_mask])
                pred_masks.append(rle_list)

        prediction.update({'prediction_masks': pred_masks})
        results.append(prediction)

    tmpdir = './dist_test_temp_res_' + args.dataset + args.split + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        metric = dataset.evaluate(results, './work_dirs')
        print(metric)


if __name__ == '__main__':
    main()
