import argparse
import re
import math
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

# 把项目根目录加到 PYTHONPATH
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
import json
from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from safetensors.torch import load_file  # ★ 你在 load_rl_state_dict_safetensors 里会用到
def parse_args():
    parser = argparse.ArgumentParser(description='RefCocog region caption')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--output-path',
        default='./region_cap_pred.json',
        help='save path of the prediction')
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

class RegionCapInferenceDataset:
    def __init__(self,
                 image_folder,
                 annotation_file=None,
                 ):
        self.image_folder = image_folder
        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())

    def __len__(self):
        return len(self.image_dict_keys)

    def decode_mask(self, annotation, image_info):
        flag = False
        masks = []

        for ann_id in range(1):

            ann = {"segmentation": annotation}

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = _mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = _mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
        masks = np.stack(masks, axis=0)

        return masks

    def get_questions(self):
        # question = "<image>\nPlease give me a short description of the region in the picture marked by region1. Please response in a word."
        question = "<image>\nPlease give me a short description of the region in the picture marked by region1."
        return question

    def __getitem__(self, index):

        data_dict = {}

        image_id = self.image_dict_keys[index]
        image_file = self.image_dict[image_id]['file_name']

        questions = self.get_questions()

        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        masks = self.ann_dict[image_id]['segmentation']
        image_info = self.image_dict[image_id]
        masks = self.decode_mask(masks, image_info)

        data_dict['image'] = image
        data_dict['text'] = questions
        data_dict['img_id'] = image_id
        data_dict['mask_prompts'] = [masks]

        return data_dict


ANNOTATION_FILE = './data/region_caption/refcocog/finetune_refcocog_val_with_mask.json'
IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'

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
    MAX_COMPLETION_LENGTH = 256
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

    dataset = RegionCapInferenceDataset(
        image_folder=IMAGE_FOLDER,
        annotation_file=ANNOTATION_FILE,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        result_dict = {'image_id': data_batch['img_id'], 'image_file': data_batch['image_file']}
        del data_batch['img_id'], data_batch['image_file']

        prediction = model.predict_forward(**data_batch, tokenizer=tokenizer)['prediction']

        text_output = prediction.replace("<s>", "").replace("\n", "") \
            .replace("region1", '').replace("Region1", '').replace("The region marked by", "").replace("The region marked as", "").replace("The region marked", "") \
            .replace("is", "").replace("shows", "").replace(':', '').replace("   ", " ").replace("  ", " ")
        text_output = text_output.split("ASSISTANT: ")[-1]
        cleaned_str = re.sub(r'<.*?>', '', text_output)
        cleaned_str = cleaned_str.replace('[SEG]', '')
        cleaned_str = ' '.join(cleaned_str.split()).strip("'")
        cleaned_str = cleaned_str.strip()

        result_dict["caption"] = cleaned_str
        result_dict["prediction"] = cleaned_str
        results.append(result_dict)

    tmpdir = './dist_test_temp_regioncap_' + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        with open(args.output_path, 'w') as json_file:
            json.dump(results, json_file, indent=2)

if __name__ == '__main__':
    main()
