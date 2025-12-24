"""
Tokenization and template functions for Sa2VA RL training.
MUST be consistent with Sa2VA SFT training.

Reference:
- projects/llava_sam2/datasets/encode_fn.py (video_lisa_encode_fn)
- pretrain_hf/templates.py (PROMPT_TEMPLATE.vicuna)
"""

import copy
from typing import Dict, Any, List
from xtuner.dataset.utils import get_bos_eos_token_ids
from xtuner.utils import IGNORE_INDEX


# Vicuna template (same as SFT training)
VICUNA_TEMPLATE = dict(
    SYSTEM=('A chat between a curious user and an artificial '
            'intelligence assistant. The assistant gives '
            'helpful, detailed, and polite answers to the '
            'user\'s questions. {system}\n '),
    INSTRUCTION=('USER: {input} ASSISTANT:'),
    SEP='\n'
)


def apply_vicuna_template(conversation: List[Dict[str, str]], system: str = '') -> List[Dict[str, str]]:
    """
    Apply Vicuna template to conversation pairs.

    Args:
        conversation: List of {'input': str, 'output': str} dicts
        system: System prompt (usually empty for Sa2VA)

    Returns:
        Templated conversation with same format
    """
    templated_conversation = []

    for i, turn in enumerate(conversation):
        # First turn may include system prompt
        if i == 0 and system:
            system_text = VICUNA_TEMPLATE['SYSTEM'].format(system=system)
            input_text = system_text + VICUNA_TEMPLATE['INSTRUCTION'].format(input=turn['input'])
        else:
            input_text = VICUNA_TEMPLATE['INSTRUCTION'].format(input=turn['input'])

        # For multi-turn, add separator between turns
        if i > 0:
            input_text = VICUNA_TEMPLATE['SEP'] + input_text

        templated_conversation.append({
            'input': input_text,
            'output': turn['output']
        })

    return templated_conversation


def video_lisa_encode_fn(
        example: Dict[str, Any],
        tokenizer,
        max_length: int,
        input_ids_with_output: bool = True,
        **kwargs
) -> Dict[str, List[int]]:
    """
    Tokenize conversation for Sa2VA model.
    Exactly the same as SFT training encode function.

    Args:
        example: Dict containing 'conversation' key
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        input_ids_with_output: Whether to include output in input_ids

    Returns:
        Dict with 'input_ids' and 'labels'
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True

    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        input_encode = tokenizer.encode(input, add_special_tokens=False)

        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)

        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)

        if input_ids_with_output:
            # Add output
            output_with_loss = single_turn_conversation.get('output_with_loss', True)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode

            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)

            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False

            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {'input_ids': input_ids, 'labels': labels}


class Sa2VATemplateAndTokenizer:
    """
    Wrapper that applies template and tokenization.
    Consistent with Sa2VA SFT training pipeline.
    """

    def __init__(self, tokenizer, max_length: int = 8196):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        preprocessed_data: Dict[str, Any],
        apply_template: bool = True,
        tokenize: bool = True
    ) -> Dict[str, Any]:
        """
        Apply template and tokenization to preprocessed data.

        Args:
            preprocessed_data: Output from Sa2VADataPreprocessor.prepare_for_model
                Must contain:
                - 'conversation': list of {'input': str, 'output': str}
                - 'pixel_values': tensor
                - 'prompt_masks': tensor
                - etc.
            apply_template: Whether to apply vicuna template
            tokenize: Whether to tokenize

        Returns:
            Updated data dict with tokenized fields
        """
        data_dict = preprocessed_data.copy()

        # Apply template
        if apply_template and 'conversation' in data_dict:
            conversation = data_dict['conversation']
            templated_conversation = apply_vicuna_template(conversation, system='')
            data_dict['conversation'] = templated_conversation

        # Tokenize
        if tokenize and 'conversation' in data_dict:
            tokenized = video_lisa_encode_fn(
                data_dict,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                input_ids_with_output=True
            )
            data_dict.update(tokenized)

        return data_dict


def prepare_sa2va_rl_input(
    image,
    mask,
    caption: str,
    preprocessor,
    tokenizer,
    max_length: int = 8196,
    task: str = "mask_to_caption",
    instruction: str = "Please generate a detailed description for the given image region."
) -> Dict[str, Any]:
    """
    Complete pipeline: raw data -> preprocessed -> templated -> tokenized.
    This is the main function to use for RL training data preparation.

    Args:
        image: PIL Image
        mask: numpy array mask
        caption: ground truth caption
        preprocessor: Sa2VADataPreprocessor instance
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        task: "mask_to_caption" or "caption_to_mask"
        instruction: instruction template

    Returns:
        Complete model input dict with:
        - pixel_values: (1, 3, H, W)
        - prompt_masks: (1, G, G)
        - vp_overall_mask: (1,)
        - prompt_text: str (for generation, not tokenized)
        - conversation: list of {'input': str, 'output': str} (templated)
        - input_ids: list of int (tokenized)
        - labels: list of int (tokenized)
        - gt_caption: str
    """
    # Step 1: Preprocess (image, mask, text formatting)
    preprocessed = preprocessor.prepare_for_model(
        image=image,
        mask=mask,
        caption=caption,
        instruction=instruction,
        task=task
    )

    # Step 2: Build conversation format (for tokenization)
    # For RL training, we create conversation pairs
    if task == "mask_to_caption":
        conversation = [{
            'input': preprocessed['prompt_text'],
            'output': caption
        }]
    else:  # caption_to_mask
        # For caption->mask, the output would be a special token indicating mask
        conversation = [{
            'input': preprocessed['prompt_text'],
            'output': '[SEG]'  # Placeholder for mask output
        }]

    preprocessed['conversation'] = conversation

    # Step 3: Apply template and tokenize
    template_tokenizer = Sa2VATemplateAndTokenizer(tokenizer, max_length)
    final_data = template_tokenizer(preprocessed, apply_template=True, tokenize=True)

    return final_data
