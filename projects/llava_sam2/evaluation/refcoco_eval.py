import argparse
import copy
import math
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
import random

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset


def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
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
    # 挤掉可能的单通道维度
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


def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
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
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))

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
                # 只用第一张（保持与你原逻辑一致）；若想融合多张可改为 OR
                first_mask = to_numpy_mask(pred_mask[0])
                rle_list = mask_to_rle([first_mask])  # 注意包成列表
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
