# scripts/merge_hf_safetensors.py
import os, glob, json
import torch
from safetensors.torch import load_file


HF_DIR = os.environ.get("HF_SA2VA_4B_DIR", "/path/to/hf_models/Sa2VA-4B")
INDEX = os.path.join(HF_DIR, "model.safetensors.index.json")

def load_all_shards(hf_dir):
    index_file = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)
        shard_map = index["weight_map"].values()
        shard_paths = sorted(set(os.path.join(hf_dir, p) for p in shard_map))
    else:
        # 单文件情况
        shard_paths = glob.glob(os.path.join(hf_dir, "*.safetensors"))
        if len(shard_paths) == 0:
            raise FileNotFoundError("No *.safetensors found in HF dir")

    merged = {}
    for p in shard_paths:
        part = load_file(p)
        for k, v in part.items():
            if k in merged:
                raise KeyError(f"duplicate key in shards: {k}")
            merged[k] = v
    return merged

if __name__ == "__main__":
    sd = load_all_shards(HF_DIR)
    print(f"Loaded {len(sd)} tensors from HF dir: {HF_DIR}")
    # 仅演示；不保存到磁盘，这个 merged 会在下一步转换时使用
