"""
Quick test to verify the tokenization fix generates correct number of <IMG_CONTEXT> tokens.
"""

import sys
sys.path.insert(0, '/data/xiaoyicheng/Sa2VA')

import torch
import pyarrow as pa
from PIL import Image
import io
from transformers import AutoTokenizer
from projects.llava_sam2.rl_train.data_preprocessor import Sa2VADataPreprocessor
import pycocotools.mask as mask_util

print("="*80)
print("Testing Tokenization Fix")
print("="*80)

# Load tokenizer
print("\n[1] Loading tokenizer...")
tokenizer_path = "/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
print(f"<IMG_CONTEXT> token ID: {img_context_token_id}")

# Load one sample
print("\n[2] Loading sample from GAR dataset...")
arrow_file = '/data/xiaoyicheng/Sa2VA/data/GAR/Fine-Grained-Dataset-Part1/data-00000-of-00044.arrow'

with pa.memory_map(arrow_file, 'r') as source:
    try:
        reader = pa.ipc.open_file(source)
        table = reader.read_all()
    except:
        source.seek(0)
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()

row = table.slice(0, 1).to_pydict()
image_bytes = row['image'][0]['bytes']
image = Image.open(io.BytesIO(image_bytes))
mask_rle = row['mask_rle'][0]

# Decode mask
mask_rle_obj = {'counts': mask_rle['counts'], 'size': mask_rle['size']}
mask = mask_util.decode(mask_rle_obj)

print(f"Image size: {image.size}")
print(f"Mask shape: {mask.shape}")

# Preprocess
print("\n[3] Preprocessing data...")
preprocessor = Sa2VADataPreprocessor()
preprocessed = preprocessor.prepare_for_model(
    image=image,
    mask=mask,
    caption="Test caption",
    task="mask_to_caption"
)

prompt_text = preprocessed['prompt_text']
region_pixels = preprocessed['region_pixels']

print(f"Region pixels: {region_pixels}")
print(f"Prompt text (before image token replacement):")
print(prompt_text[:500])

# Apply image token replacement (the FIX)
print("\n[4] Applying image token replacement...")
IMG_TOKENS_PER_FRAME = 256
num_image_tokens = 1 * IMG_TOKENS_PER_FRAME
image_token_str = f"<img>{'<IMG_CONTEXT>' * num_image_tokens}</img>"
prompt_text_fixed = prompt_text.replace('<image>', image_token_str)

print(f"Image token string length: {len(image_token_str)} chars")
print(f"Prompt text (after image token replacement) length: {len(prompt_text_fixed)} chars")

# Tokenize
print("\n[5] Tokenizing...")
tokens = tokenizer(prompt_text_fixed, return_tensors='pt')
input_ids = tokens['input_ids'][0]

# Count <IMG_CONTEXT> tokens
num_img_context = (input_ids == img_context_token_id).sum().item()

print(f"Tokenized length: {len(input_ids)}")
print(f"Number of <IMG_CONTEXT> tokens: {num_img_context}")

# Expected count
expected_count = IMG_TOKENS_PER_FRAME + sum(region_pixels)
print(f"\n[6] Verification:")
print(f"Expected <IMG_CONTEXT> count: {IMG_TOKENS_PER_FRAME} (base) + {sum(region_pixels)} (VPs) = {expected_count}")
print(f"Actual <IMG_CONTEXT> count: {num_img_context}")

if num_img_context == expected_count:
    print("\n✓ SUCCESS: Token count matches expected!")
else:
    print(f"\n✗ MISMATCH: Expected {expected_count} but got {num_img_context}")
    print(f"Difference: {num_img_context - expected_count}")

# Also check if the model's prepare_inputs_embeds would work
print("\n[7] Estimating vp_embeds size...")
# Based on modeling_sa2va_chat.py:
# vp_embeds = full VIT embeds (1024) + VP-selected embeds (region_pixels)
# But wait, the SFT code uses 256 tokens, so maybe the model downsamples?
# Let's just report what we know
print(f"VIT tokens per image: 1024 (32x32 patches)")
print(f"LLM tokens per image (after downsampling): 256 (16x16 grid)")
print(f"VP tokens for this sample: {sum(region_pixels)}")
print(f"Total expected tokens: 256 + {sum(region_pixels)} = {256 + sum(region_pixels)}")

print("\n" + "="*80)
print("Test complete!")
print("="*80)
