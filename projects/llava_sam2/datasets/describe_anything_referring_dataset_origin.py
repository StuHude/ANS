# projects/llava_sam2/datasets/describe_anything_referring_dataset.py
# -*- coding: utf-8 -*-

import os
import io
import pickle as pkl
import logging
from typing import List, Tuple, Union, Optional, Dict, Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_tensor as pil_to_tensor
from PIL import Image

from datasets import (
    load_dataset,
    get_dataset_config_names,
    DownloadConfig,
)

from xtuner.registry import BUILDER

logger = logging.getLogger(__name__)

# ------- 缺省视觉前处理（与 InternVL 对齐） -------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_IMAGE_SIZE = 448
PATCH_SIZE = 14
DOWNSAMPLE_RATIO = 0.5
# token 网格边长 G 与每帧 token 数
GRID_SIZE = int((INTERNVL_IMAGE_SIZE // PATCH_SIZE) * DOWNSAMPLE_RATIO)  # 16
IMG_TOKENS_PER_FRAME = GRID_SIZE * GRID_SIZE                             # 256
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def _ensure_rel_data_dir(module_dir: str, abs_data_root: Optional[str]) -> Optional[str]:
    if abs_data_root is None:
        return None
    rel = "_data"
    target = os.path.join(module_dir, rel)
    try:
        if not (os.path.islink(target) or os.path.exists(target)):
            os.symlink(abs_data_root, target)
    except Exception as e:
        logger.warning(f"Create symlink failed: {target} -> {abs_data_root}: {e}")
    return rel


def _decode_rle_to_mask(mask_rle: Dict[str, Any]) -> torch.Tensor:
    """HF/COCO RLE -> (1,H,W) float {0,1}"""
    try:
        from pycocotools import mask as cocomask
    except Exception as e:
        raise RuntimeError("需要 pycocotools 才能解码 RLE：pip install pycocotools") from e

    rle = dict(mask_rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    m = cocomask.decode(rle)  # (H,W) 或 (H,W,N)
    if m.ndim == 3:
        m = m[..., 0]
    m = torch.from_numpy(m.astype("float32"))  # (H,W)
    m = (m > 0.5).float().unsqueeze(0)  # (1,H,W)
    return m


def _wrap_image_processor(proc: Any) -> Callable[[Image.Image], torch.Tensor]:
    if callable(proc):
        def _call(img: Image.Image) -> torch.Tensor:
            out = proc(img)
            if isinstance(out, Image.Image):
                out = transforms.ToTensor()(out)
            elif not isinstance(out, torch.Tensor):
                out = transforms.ToTensor()(out)
            return out
        return _call

    for m in ("process", "forward", "transform", "resize"):
        if hasattr(proc, m):
            fn = getattr(proc, m)
            def _call(img: Image.Image, _fn=fn) -> torch.Tensor:
                out = _fn(img)
                if isinstance(out, Image.Image):
                    out = transforms.ToTensor()(out)
                elif not isinstance(out, torch.Tensor):
                    out = transforms.ToTensor()(out)
                return out
            return _call

    tfm = transforms.Compose([
        transforms.Resize((INTERNVL_IMAGE_SIZE, INTERNVL_IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def _call(img: Image.Image, _tfm=tfm) -> torch.Tensor:
        return _tfm(img)

    return _call


# --------- meta 中 region 列表抽取 & 有效性判定 ---------
def _as_region_list(meta_obj: Any) -> List[dict]:
    if isinstance(meta_obj, list):
        return meta_obj
    if isinstance(meta_obj, dict):
        for k in ("regions", "items", "objects", "annotations", "list"):
            v = meta_obj.get(k, None)
            if isinstance(v, list):
                return v
        vals = list(meta_obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals
    return []


def _is_valid_region(it: Any) -> bool:
    if not isinstance(it, dict):
        return False
    cap = it.get("caption", None)
    mrl = it.get("mask_rle", None)
    if not (isinstance(cap, str) and cap.strip()):
        return False
    if not (isinstance(mrl, dict) and ("size" in mrl) and ("counts" in mrl)):
        return False
    return True


@BUILDER.register_module()
class DescribeAnythingReferringDataset(Dataset):
    """
    只做 “image + mask(visual prompt) → caption (CE)”：
      - 输出给 MLLM：pixel_values, input_ids, labels
      - 提供 visual prompt：prompt_masks (list[Tensor(bool, shape=[N_tokens])]) + vp_overall_mask (Tensor[n_frames] 的 bool)
      - 不输出 g_pixel_values / masks（避免触发 SAM 分割损失）
    """

    def __init__(self,
                 hf_dataset_name: str,
                 hf_dataset_config: Union[str, List[str], Tuple[str, ...]],
                 tokenizer,
                 template_map_fn,
                 extra_image_processor,
                 max_length: int,
                 instruction_template: str,
                 local_data_root: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 repeats: int = 1,
                 lazy: bool = True,
                 hf_split: str = "train",
                 dataset_repo_dir: Optional[str] = None,
                 fast_pair_sampling: bool = True,
                 precomputed_total_pairs: Optional[int] = None,
                 **_ignore_kwargs) -> None:

        # 组件
        self.tokenizer = BUILDER.build(tokenizer) if isinstance(tokenizer, dict) else tokenizer
        self.template_map_fn = template_map_fn
        self._raw_image_processor = BUILDER.build(extra_image_processor) \
            if isinstance(extra_image_processor, dict) else extra_image_processor
        self._image_processor = _wrap_image_processor(self._raw_image_processor)

        self.max_length = int(max_length)
        self.instruction_template = str(instruction_template).strip()
        self.repeats = int(repeats)
        self.lazy = bool(lazy)
        self.max_tiles_upper_bound = 96

        # config 规范化
        if isinstance(hf_dataset_config, (list, tuple)):
            cfgs = list(dict.fromkeys(hf_dataset_config))
        else:
            cfgs = [hf_dataset_config]

        # 路径解析
        resolve_paths: List[Tuple[str, str, bool]] = []
        if dataset_repo_dir is not None:
            for c in cfgs:
                subdir = os.path.join(dataset_repo_dir, c)
                resolve_paths.append((c, subdir, True))
        else:
            for c in cfgs:
                resolve_paths.append((c, hf_dataset_name, False))

        # 校验
        if dataset_repo_dir is not None:
            bad = [c for (c, p, _) in resolve_paths if not os.path.isdir(p)]
            if bad:
                raise ValueError(f"以下子配置在本地脚本仓库中未找到：{bad}；请检查 dataset_repo_dir='{dataset_repo_dir}'。")
        else:
            available = set(get_dataset_config_names(hf_dataset_name))
            bad = [c for c in cfgs if c not in available]
            if bad:
                raise ValueError(f"无效的 hf_dataset_config: {bad}；可用配置：{sorted(available)}")

        logger.info(
            "DescribeAnythingReferringDataset | "
            f"dataset_repo_dir={dataset_repo_dir}, "
            f"hf_dataset_name={hf_dataset_name}, "
            f"configs={cfgs}, split='{hf_split}', data_dir='{local_data_root}', cache_dir='{cache_dir}', "
            f"lazy={self.lazy}, fast_pair_sampling={fast_pair_sampling}, precomputed_total_pairs={precomputed_total_pairs}"
        )

        # 加载 split（不解码）
        self.subsets: List[object] = []
        for (name, path, is_local_script) in [(n, p, l) for (n, p, l) in resolve_paths]:
            if is_local_script:
                rel_data_dir = _ensure_rel_data_dir(path, local_data_root)
                kwargs = dict(split=hf_split, cache_dir=cache_dir,
                              download_config=DownloadConfig(local_files_only=True))
                if rel_data_dir is not None:
                    kwargs["data_dir"] = rel_data_dir
                ds_split = load_dataset(path, **kwargs)
            else:
                ds_split = load_dataset(path, name=name, split=hf_split,
                                        data_dir=local_data_root, cache_dir=cache_dir)
            self.subsets.append(ds_split)

        # 两种模式：fast（惰性） vs full（逐 region）
        self.fast_pair_sampling = bool(fast_pair_sampling)
        if self.fast_pair_sampling:
            self.records: List[Tuple[int, int]] = []  # (subset_id, raw_idx)
            for subset_id, ds_split in enumerate(self.subsets):
                self.records.extend((subset_id, i) for i in range(len(ds_split)))

            if precomputed_total_pairs is not None:
                self._fast_total_pairs = int(precomputed_total_pairs)
            else:
                self._fast_total_pairs = max(1, len(self.records) * 2)

            self._mode = "fast"
            self.total_len = self._fast_total_pairs
        else:
            self.samples: List[Tuple[int, int, int]] = []
            for subset_id, ds_split in enumerate(self.subsets):
                for i in range(len(ds_split)):
                    rec = ds_split[i]
                    pb = rec.get('pickle', None)
                    if pb is None:
                        continue
                    try:
                        meta = pkl.loads(pb)
                    except Exception:
                        continue
                    region_list = _as_region_list(meta)
                    for j, it in enumerate(region_list):
                        if _is_valid_region(it):
                            self.samples.append((subset_id, i, j))
            self._mode = "full"
            self.total_len = len(self.samples)

        # 给 LengthGroupedSampler：固定长度数组
        self.modality_length = [self.max_length] * max(1, self.total_len)

        # 汇总
        effective_len = self.total_len * self.repeats
        msg_header = (f"Dataset ready => mode={self._mode}, subsets={len(self.subsets)}, "
                      f"samples(pairs)={self.total_len}, repeats={self.repeats}, "
                      f"effective_len={effective_len}, split='{hf_split}'")
        logger.info(msg_header)
        print(msg_header, flush=True)

    def __len__(self) -> int:
        return self.total_len * self.repeats

    # ---- 工具 ----
    def _load_pil_from_record(self, rec: Dict[str, Any]) -> Image.Image:
        jpg_obj = rec['jpg']
        if isinstance(jpg_obj, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(jpg_obj)).convert("RGB")
        if isinstance(jpg_obj, Image.Image):
            return jpg_obj.convert("RGB") if jpg_obj.mode != "RGB" else jpg_obj
        try:
            from torchvision.transforms.functional import to_pil_image
            return to_pil_image(jpg_obj).convert("RGB")
        except Exception as e:
            raise TypeError(f"'jpg' 字段既不是 bytes 也不是 PIL.Image，且无法 to_pil：type={type(jpg_obj)}") from e

    def _apply_image_processor(self, img: Image.Image) -> torch.Tensor:
        out = self._image_processor(img)
        if not isinstance(out, torch.Tensor):
            out = transforms.ToTensor()(out)
        if out.ndim != 3:
            raise RuntimeError(f"extra_image_processor 应输出 (C,H,W) 张量，当前 shape={getattr(out, 'shape', None)}")
        return out

    def _map_fast_index_to_record_and_region(self, idx: int) -> Tuple[int, int, int]:
        base_len = len(self.records)
        base_i = idx % base_len
        round_i = idx // base_len
        subset_id, raw_idx = self.records[base_i]

        rec = self.subsets[subset_id][raw_idx]
        meta_obj = pkl.loads(rec['pickle'])
        region_list = _as_region_list(meta_obj)
        valid = [j for j, it in enumerate(region_list) if _is_valid_region(it)]
        if not valid:
            for step in range(1, base_len + 1):
                subset_id2, raw_idx2 = self.records[(base_i + step) % base_len]
                rec2 = self.subsets[subset_id2][raw_idx2]
                meta2 = pkl.loads(rec2['pickle'])
                region_list2 = _as_region_list(meta2)
                valid2 = [j for j, it in enumerate(region_list2) if _is_valid_region(it)]
                if valid2:
                    return subset_id2, raw_idx2, valid2[round_i % len(valid2)]
            return subset_id, raw_idx, 0

        region_idx = valid[round_i % len(valid)]
        return subset_id, raw_idx, region_idx

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1) 选样本与 region
        if self.fast_pair_sampling:
            real_idx = idx % self.total_len
            subset_id, raw_idx, region_idx = self._map_fast_index_to_record_and_region(real_idx)
            rec = self.subsets[subset_id][raw_idx]
            meta_obj = pkl.loads(rec['pickle'])
            region_list = _as_region_list(meta_obj)
            item = region_list[region_idx]
            if not _is_valid_region(item):
                return self.__getitem__(idx + 1)
        else:
            real_idx = idx % self.total_len
            subset_id, raw_idx, region_idx = self.samples[real_idx]
            rec = self.subsets[subset_id][raw_idx]
            meta_obj = pkl.loads(rec['pickle'])
            region_list = _as_region_list(meta_obj)
            if not (0 <= region_idx < len(region_list)) or not _is_valid_region(region_list[region_idx]):
                return self.__getitem__(idx + 1)
            item = region_list[region_idx]

        # 2) 图像（给 MLLM）
        img = self._load_pil_from_record(rec)
        pixel_1 = self._apply_image_processor(img)                       # (3,448,448) 归一化
        pixel_values = torch.stack([pixel_1.contiguous()], dim=0)        # (1,3,448,448)

        # 3) region mask → visual prompt（仅给 MLLM，用作 prompt，不做 seg 监督）
        caption: str = str(item.get('caption', '')).strip()
        mask_rle: Dict[str, Any] = item['mask_rle']
        mask_1hw = _decode_rle_to_mask(mask_rle)  # (1,H0,W0)

        # 3.1 对齐到视觉输入尺寸
        mask_resizer = transforms.Resize(pixel_1.shape[-2:], interpolation=InterpolationMode.NEAREST)
        mask_1hw = mask_resizer(mask_1hw)  # (1,H,W) float in {0,1}

        # 3.2 聚合到 token 网格 G×G（与现有数据集保持一致：AvgPool + 0.5 阈值 + 展平）
        pooled_1gg = F.adaptive_avg_pool2d(mask_1hw, (GRID_SIZE, GRID_SIZE))  # (1, G, G)
        prob_gg = pooled_1gg[0]                                               # (G, G) in [0,1]
        token_prob_1d = prob_gg.flatten()                                     # (256,)
        token_bool_1d = token_prob_1d > 0.5                                   # (256,)

        # —— 只给“单行”掩码：(1, 256) —— 
        # 这样 default_collate 会堆成 (B, 1, 256)，模型端每个样本只会看到 1 行，避免 T×16 溢出。
        K_PER_TILE = 1
        #K_PER_TILE = max(1, 256 // self.max_tiles_upper_bound)  # 96 → 2
        #K_PER_TILE = min(K_PER_TILE, 4)  # 最多 4；按上界 96 仍然是 2

        true_idx = torch.nonzero(token_bool_1d, as_tuple=False).flatten()

        if true_idx.numel() == 0:
            topk = min(K_PER_TILE, token_prob_1d.numel())
            sel_cols = torch.topk(token_prob_1d, k=topk, dim=0).indices  # (K_PER_TILE,)
        else:
            if true_idx.numel() >= K_PER_TILE:
                step = true_idx.numel() / float(K_PER_TILE)
                sel = torch.round(torch.arange(0, K_PER_TILE, dtype=torch.float32) * step).long()
                sel = torch.clamp(sel, max=true_idx.numel() - 1)
                sel_cols = true_idx[sel]                                   # (K_PER_TILE,)
            else:
                rep = (K_PER_TILE + true_idx.numel() - 1) // true_idx.numel()
                sel_cols = true_idx.repeat(rep)[:K_PER_TILE]               # (K_PER_TILE,)

        row_mask = torch.zeros(IMG_TOKENS_PER_FRAME, dtype=torch.bool)     # (256,)
        row_mask[sel_cols] = True
        prompt_masks = row_mask.unsqueeze(0)                                # ★ 形状变为 (1,256)
        vp_overall_mask = torch.tensor([False], dtype=torch.bool)



        # 4) 注入 <IMG_CONTEXT> 占位 + 指令 → CE
        img_ctx_str = IMG_CONTEXT_TOKEN * IMG_TOKENS_PER_FRAME
        conversation = [
            {'from': 'user', 'value': img_ctx_str + '\n' + self.instruction_template},
            {'from': 'assistant', 'value': caption}
        ]

        # 5) 文本打包（优先 template_map_fn，再兜底）
        processed: Dict[str, Any] = {}
        for keyname in ("conversation", "messages"):
            try:
                out = self.template_map_fn({keyname: conversation})
                if isinstance(out, dict) and 'input_ids' in out:
                    processed = out
                    break
            except Exception:
                pass

        if 'input_ids' not in processed:
            ids = None
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    conv_msgs = []
                    for turn in conversation:
                        role = turn.get('from')
                        text = turn.get('value', '')
                        if role == 'user':
                            conv_msgs.append({"role": "user", "content": text})
                        elif role == 'assistant':
                            conv_msgs.append({"role": "assistant", "content": text})
                        else:
                            conv_msgs.append({"role": "user", "content": text})
                    ids = self.tokenizer.apply_chat_template(
                        conv_msgs, tokenize=True, add_generation_prompt=False
                    )
                    if hasattr(ids, "tolist"):
                        ids = ids.tolist()
                except Exception:
                    ids = None
            if ids is None:
                joined = f"User: {img_ctx_str}\n{self.instruction_template}\nAssistant: {caption}"
                ids = self.tokenizer(joined, add_special_tokens=True)["input_ids"]
            processed = {"input_ids": ids, "labels": list(ids)}

        # 6) 截断
        input_ids = processed['input_ids']
        labels = processed.get('labels', list(input_ids))
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        processed['input_ids'] = input_ids
        processed['labels'] = labels

        # 7) 按 collate 识别的字段组织输出（只给 MLLM & VP）
        processed['pixel_values'] = pixel_values                  # (1,3,448,448)
        processed['prompt_masks'] = prompt_masks                  # [ (256,) bool ]
        processed['vp_overall_mask'] = vp_overall_mask           # (1,) bool（单帧 True）
        # 不提供 g_pixel_values / masks：避免分割监督

        # 组 batch 的长度提示
        processed['modality_length'] = self.max_length
        # 预算 = 256（每个 <IMG_CONTEXT>）* (每行 True 个数) + overall(True 的个数)
        k_per_row = int(prompt_masks.sum().item())              # 现在应该是 1
        overall = int(vp_overall_mask.sum().item()) if 'vp_overall_mask' in locals() else 0
        assert k_per_row * IMG_TOKENS_PER_FRAME + overall <= IMG_TOKENS_PER_FRAME, \
           f"VP over budget: {k_per_row * IMG_TOKENS_PER_FRAME + overall} > {IMG_TOKENS_PER_FRAME}; " \
           f"set vp_overall_mask=False or reduce K."

        return processed
