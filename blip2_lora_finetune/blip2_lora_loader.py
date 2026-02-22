from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    BlipImageProcessor,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from lora_settings import get_lora_settings
import logging
import os


def _resolve_lora_targets(model):
    """Resolve LoRA target module names that exist in the current model."""
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "query", "key", "value",
        "q", "k", "v", "o",
    ]
    names = {name for name, _ in model.named_modules()}
    matches = []
    for cand in candidates:
        if any(name.endswith(cand) for name in names):
            matches.append(cand)
    return matches

logger = logging.getLogger(__name__)


def load_lora_blip2(
    model_id=None,
    tokenizer_path=None,
    image_processor_path=None,
):
    model_id = model_id or os.getenv("BLIP2_BASE_MODEL", "Salesforce/blip2-opt-2.7b")
    tokenizer_path = tokenizer_path or os.getenv("BLIP2_TOKENIZER_PATH", model_id)
    image_processor_path = image_processor_path or os.getenv("BLIP2_IMAGE_PROCESSOR_PATH", model_id)
    local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "0") == "1"

    # Load tokenizer and image processor.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=local_files_only
    )
    image_processor = BlipImageProcessor.from_pretrained(
        image_processor_path, local_files_only=local_files_only
    )
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

    logger.info("Tokenizer loaded: %s", type(processor.tokenizer).__name__)
    logger.info("Tokenizer vocab size: %s", len(processor.tokenizer))
    logger.info(
        "Special token IDs - EOS: %s, BOS: %s, PAD: %s",
        processor.tokenizer.eos_token_id,
        processor.tokenizer.bos_token_id,
        processor.tokenizer.pad_token_id,
    )

    # Load the base BLIP-2 model and attach LoRA adapters.
    load_in_8bit = os.getenv("LOAD_IN_8BIT", "1") == "1"
    model_kwargs = {"device_map": "auto"}
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        **model_kwargs,
    )

    if getattr(processor, "num_query_tokens", None) is None:
        processor.num_query_tokens = getattr(model.config, "num_query_tokens", 32)
        logger.info("Processor num_query_tokens set to: %s", processor.num_query_tokens)

    model = prepare_model_for_kbit_training(model)

    lora_cfg = get_lora_settings()
    if not lora_cfg.target_modules:
        lora_cfg.target_modules = _resolve_lora_targets(model)
    else:
        available = set(_resolve_lora_targets(model))
        requested = list(lora_cfg.target_modules)
        valid = [m for m in requested if m in available]
        if not valid:
            valid = sorted(available)
        lora_cfg.target_modules = valid

    if not lora_cfg.target_modules:
        raise ValueError("No LoRA target modules detected for this model.")

    logger.info("LoRA target modules: %s", lora_cfg.target_modules)
    model = get_peft_model(model, lora_cfg)

    return model, processor
