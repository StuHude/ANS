import argparse
import copy
import os.path as osp
import os
import sys
import json
import torch
import numpy as np
from mmengine.dist import master_only
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
from collections.abc import Mapping, Sequence

# 兼容你本地的 sys.path 使用方式
sys.path.insert(0, "/data/xiaoyicheng/Sa2VA/")

def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth-model', help='pth model file')
    parser.add_argument('--save-path', type=str, default='./work_dirs/hf_model',
                        help='save folder name')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)

def _jsonable(obj):
    """将 config 字典转换为可 JSON 序列化的结构（最小清洗）"""
    # 基础类型
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy 标量 / 数组
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # torch 相关
    if isinstance(obj, torch.dtype):
        return str(obj).replace('torch.', '')  # 例如 'bfloat16'
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.Size):
        return list(obj)

    # 容器
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]

    # 其它（比如函数、类等），退化为字符串
    if callable(obj):
        return f"<callable:{getattr(obj, '__name__', 'anonymous')}>"

    return str(obj)

def main():
    args = parse_args()

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)
    backend = get_file_backend(args.pth_model)

    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    # 合并 LoRA，切到导出模式
    model._merge_lora()
    model.mllm.transfer_to_hf = True

    # 导出权重（保持原逻辑）
    all_state_dict = model.all_state_dict()
    name_map = {'mllm.model.': '', '.gamma': '.g_weight'}
    all_state_dict_new = {}
    for key in all_state_dict.keys():
        new_key = copy.deepcopy(key)
        for _text in name_map.keys():
            new_key = new_key.replace(_text, name_map[_text])
        all_state_dict_new[new_key] = all_state_dict[key]

    # 构建 HF 配置与模型（保持原逻辑）
    from projects.llava_sam2.hf.models.configuration_sa2va_chat import Sa2VAChatConfig
    from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel

    internvl_config = Sa2VAChatConfig.from_pretrained(cfg.path)
    config_dict = internvl_config.to_dict()

    config_dict['auto_map'] = {
        'AutoConfig': 'configuration_sa2va_chat.Sa2VAChatConfig',
        'AutoModel': 'modeling_sa2va_chat.Sa2VAChatModel',
        'AutoModelForCausalLM': 'modeling_sa2va_chat.Sa2VAChatModel'
    }

    # 显式补齐 llm_config，避免项目里的 __init__ 假设导致 KeyError
    config_dict.setdefault("llm_config", {})
    config_dict["llm_config"].setdefault("architectures", ["Qwen2ForCausalLM"])
    config_dict["llm_config"]["vocab_size"] = len(model.tokenizer)
    config_dict["template"] = cfg.template

    sa2va_hf_config = Sa2VAChatConfig(**config_dict)

    hf_sa2va_model = Sa2VAChatModel(
        sa2va_hf_config,
        vision_model=model.mllm.model.vision_model,
        language_model=model.mllm.model.language_model,
    )
    hf_sa2va_model.load_state_dict(all_state_dict_new)

    # =========================
    # 最小改动：手动写出 HF 目录
    # =========================
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)

    # 1) 写 config.json（先清洗再写）
    clean_cfg = _jsonable(sa2va_hf_config.to_dict())
    with open(osp.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(clean_cfg, f, ensure_ascii=False, indent=2)

    # 2) 写模型权重
    torch.save(hf_sa2va_model.state_dict(), osp.join(save_dir, "pytorch_model.bin"))

    # 3) 保存 tokenizer
    model.tokenizer.save_pretrained(save_dir)

    print(f"Save the hf model into {save_dir}")

    # 4) 复制自定义 HF 模型代码（保持原脚本行为）
    os.system(f"cp -pr ./projects/llava_sam2/hf/models/* {save_dir}")

if __name__ == '__main__':
    main()
