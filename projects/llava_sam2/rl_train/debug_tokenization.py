"""
Debug script to compare tokenization between SFT and RL training.
This will help us understand why there's a shape mismatch.
"""

import sys
sys.path.insert(0, '/data/xiaoyicheng/Sa2VA')

import torch
import pyarrow as pa
import pycocotools.mask as mask_util
from transformers import AutoTokenizer
from PIL import Image
import io

# Load a single sample from GAR dataset
arrow_file = '/data/xiaoyicheng/Sa2VA/data/GAR/Fine-Grained-Dataset-Part1/data-00000-of-00044.arrow'

print("="*80)
print("Loading sample from GAR dataset...")
print("="*80)

with pa.memory_map(arrow_file, 'r') as source:
    try:
        reader = pa.ipc.open_file(source)
        table = reader.read_all()
    except:
        source.seek(0)
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()

# Get first sample
row = table.slice(0, 1).to_pydict()

image_bytes = row['image'][0]['bytes']
image = Image.open(io.BytesIO(image_bytes))
mask_rle = row['mask_rle'][0]
conversations = row['conversations'][0]

print(f"\nImage size: {image.size}")
print(f"Conversations: {conversations}")
print(f"Number of turns: {len(conversations)}")

# Decode mask
mask_rle_obj = {'counts': mask_rle['counts'], 'size': mask_rle['size']}
mask = mask_util.decode(mask_rle_obj)  # (H, W)
print(f"Mask shape: {mask.shape}")
print(f"Mask pixels: {mask.sum()}")

# Now let's see what SFT training does with this
# Load tokenizer
print("\n" + "="*80)
print("Loading tokenizer...")
print("="*80)

tokenizer_path = "/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

# Check for special tokens
special_tokens = ['<image>', '<|im_start|>', '<|im_end|>', '<IMG_CONTEXT>']
for token in special_tokens:
    if token in tokenizer.get_vocab():
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"Token '{token}': id={token_id}")
    else:
        print(f"Token '{token}': NOT FOUND in vocabulary")

# Now let's try to build the prompt like SFT does
print("\n" + "="*80)
print("Testing SFT-style prompt construction...")
print("="*80)

# Extract user query and assistant response
user_msg = conversations[0]['value']  # User's question
assistant_msg = conversations[1]['value']  # Assistant's response

print(f"\nUser message: {user_msg}")
print(f"Assistant message: {assistant_msg}")

# SFT uses Qwen2 chat template
# Format: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>

# For Sa2VA with visual prompts, the format should be:
# <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>

system_prompt = "You are a helpful assistant."

# Build conversation
chat_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"<image>\n{user_msg}"},
    {"role": "assistant", "content": assistant_msg}
]

# Apply chat template
prompt_text = tokenizer.apply_chat_template(
    chat_messages,
    tokenize=False,
    add_generation_prompt=False
)

print(f"\nFormatted prompt:\n{prompt_text}")
print(f"\nPrompt length: {len(prompt_text)} chars")

# Tokenize
tokens = tokenizer(prompt_text, return_tensors='pt')
input_ids = tokens['input_ids'][0]

print(f"\nTokenized length: {len(input_ids)}")
print(f"First 50 token IDs: {input_ids[:50].tolist()}")
print(f"Last 50 token IDs: {input_ids[-50:].tolist()}")

# Count <IMG_CONTEXT> tokens (this is what the model looks for)
# First, find what token ID corresponds to <IMG_CONTEXT>
img_context_token = "<IMG_CONTEXT>"
if img_context_token in tokenizer.get_vocab():
    img_context_id = tokenizer.convert_tokens_to_ids(img_context_token)
    num_img_context = (input_ids == img_context_id).sum().item()
    print(f"\n<IMG_CONTEXT> token ID: {img_context_id}")
    print(f"Number of <IMG_CONTEXT> tokens: {num_img_context}")
else:
    print(f"\n<IMG_CONTEXT> token NOT in vocabulary!")
    # Check if <image> is used instead
    if '<image>' in tokenizer.get_vocab():
        image_id = tokenizer.convert_tokens_to_ids('<image>')
        num_image = (input_ids == image_id).sum().item()
        print(f"<image> token ID: {image_id}")
        print(f"Number of <image> tokens: {num_image}")

# Now let's understand visual prompt processing
print("\n" + "="*80)
print("Understanding visual prompt embedding sizes...")
print("="*80)

# Based on modeling_sa2va_chat.py lines 342-352:
# For each image with visual prompts:
# 1. vit_embeds are extracted: shape (n_img, hw, C) where hw depends on image size
# 2. For images with VPs: tile_vit_embeds are repeated for each region
# 3. Then masked by prompt_masks

# Sa2VA uses 448x448 images, vision encoder with patch_size=14
# So hw = (448/14)^2 = 32^2 = 1024

image_size = 448
patch_size = 14
num_patches = (image_size // patch_size) ** 2
print(f"Image size: {image_size}x{image_size}")
print(f"Patch size: {patch_size}")
print(f"Number of patches (hw): {num_patches}")

# With 1 image having 1 region (prompt_mask shape is (1, 16, 16)):
# - vit_embeds[0]: (1024, C)
# - tile_vit_embeds: (1024, C)
# - repeated for 1 region: (1, 1024, C)
# - prompt_mask: (1, 16, 16) -> flattened: (1, 256)
# - After masking: vp_embeds should have size = number of True values in (1, 256)

# Let's calculate expected vp_embeds size
from projects.llava_sam2.rl_train.data_preprocessor import Sa2VADataPreprocessor
import torch.nn.functional as F

preprocessor = Sa2VADataPreprocessor(image_size=448, grid_size=16)

# Process the mask
mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)
pooled = F.adaptive_avg_pool2d(mask_tensor, (16, 16))
prompt_masks_bin = (pooled > 0.5).to(torch.uint8)  # (1, 16, 16)

num_vp_tokens = prompt_masks_bin.sum().item()
print(f"\nPrompt mask shape: {prompt_masks_bin.shape}")
print(f"Number of TRUE values in prompt mask: {num_vp_tokens}")

# According to modeling_sa2va_chat.py line 343:
# For each image, FULL vit_embeds are added: vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
# Then for images with VPs, ADDITIONAL vp_embeds are added

# So total vp_embeds size for 1 image with 1 VP region:
# = num_patches (full vit embeds) + num_vp_tokens (VP-selected embeds)
# = 1024 + num_vp_tokens

expected_vp_embeds_size = num_patches + num_vp_tokens
print(f"\nExpected vp_embeds size:")
print(f"  Full VIT embeds: {num_patches}")
print(f"  + VP-selected embeds: {num_vp_tokens}")
print(f"  = Total: {expected_vp_embeds_size}")

print("\n" + "="*80)
print("CRITICAL INSIGHT:")
print("="*80)
print(f"The tokenizer must generate EXACTLY {expected_vp_embeds_size} <IMG_CONTEXT> tokens")
print(f"to match the vp_embeds size!")
print("="*80)
