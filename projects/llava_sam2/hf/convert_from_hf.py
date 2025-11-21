# projects/llava_sam2/hf/convert_from_hf.py
import os, glob, json
from datetime import datetime
from typing import Dict
import torch
from safetensors.torch import load_file

# =======【把下面两行改成你的绝对路径】=======
HF_DIR  = "/data/xiaoyicheng/Sa2VA/pretrain_hf"                  # HF 的 Sa2VA-4B 目录
OUT_PTH = "/data/xiaoyicheng/Sa2VA/pretrained/4B_checkpoint/sa2va_4b_from_hf.pth" # 产出的项目风格 .pth
# ============================================

# 按 convert_to_hf.py 反向处理：
# to_hf: 'mllm.model.' -> ''      => from_hf: (仅 MLLM 根键) '' -> 'mllm.model.'
# to_hf: '.gamma' -> '.g_weight'  => from_hf: '.g_weight' -> '.gamma'
HF_MLLM_ROOT_PREFIXES = (
    "vision_model.", "language_model.", "embed_tokens.", "lm_head."
    # 如果训练时报 mllm 大量 missing，可再补一个：
    # "model.",
)

def _load_all_hf_tensors(hf_dir: str) -> Dict[str, torch.Tensor]:
    idx = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, "r") as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(os.path.join(hf_dir, p) for p in weight_map.values()))
    else:
        shards = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No *.safetensors in {hf_dir}")
    merged = {}
    for sp in shards:
        part = load_file(sp)
        for k, v in part.items():
            if k in merged:
                raise KeyError(f"duplicate tensor key: {k}")
            merged[k] = v
    return merged

def _hf_key_to_project_key(hf_key: str) -> str:
    k = hf_key
    if ".g_weight" in k:
        k = k.replace(".g_weight", ".gamma")
    if k.startswith(HF_MLLM_ROOT_PREFIXES):
        k = "mllm.model." + k
    return k

def main():
    assert os.path.isdir(HF_DIR), f"HF dir not found: {HF_DIR}"
    os.makedirs(os.path.dirname(OUT_PTH), exist_ok=True)

    hf_sd = _load_all_hf_tensors(HF_DIR)
    proj_sd = {}
    for k, t in hf_sd.items():
        nk = _hf_key_to_project_key(k)
        if nk in proj_sd:
            i = 1
            while f"{nk}__dup{i}" in proj_sd:
                i += 1
            nk = f"{nk}__dup{i}"
        proj_sd[nk] = t

    ckpt = {"meta": {
                "version": "sa2va_4b_from_hf",
                "time": datetime.now().isoformat(),
                "note": "reverse of convert_to_hf.py name_map"},
            "state_dict": proj_sd}
    torch.save(ckpt, OUT_PTH)
    print(f"[OK] wrote checkpoint: {OUT_PTH}")
    print(f"tensors: {len(proj_sd)}")

if __name__ == "__main__":
    main()
