import json
import logging
import os
from contextlib import nullcontext

import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("inference.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _env(name, default):
    return os.getenv(name, default)


def load_model_and_processor():
    base_model_id = _env("BLIP2_BASE_MODEL", "Salesforce/blip2-opt-2.7b")
    lora_checkpoint = _env("LORA_CHECKPOINT", os.path.join("checkpoint", "checkpoint-3750"))
    processor_path = _env("BLIP2_PROCESSOR_PATH", os.path.join("checkpoint", "processor"))
    load_in_8bit = _env("LOAD_IN_8BIT", "1") == "1"

    processor_source = processor_path if os.path.exists(processor_path) else base_model_id
    logger.info("Loading processor from: %s", processor_source)
    processor = Blip2Processor.from_pretrained(processor_source)

    model_kwargs = {"device_map": "auto"}
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    logger.info("Loading base model from: %s", base_model_id)
    base_model = Blip2ForConditionalGeneration.from_pretrained(base_model_id, **model_kwargs)

    model = base_model
    if os.path.exists(lora_checkpoint):
        logger.info("Loading LoRA adapter from: %s", lora_checkpoint)
        model = PeftModel.from_pretrained(base_model, lora_checkpoint)
        model = model.merge_and_unload()
    else:
        logger.warning("LoRA checkpoint not found, using base model only: %s", lora_checkpoint)

    model.eval()
    return processor, model


def generate_caption(processor, model, image, prompt, max_new_tokens, num_beams):
    device = next(model.parameters()).device
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    use_cuda_amp = torch.cuda.is_available() and str(device).startswith("cuda")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_cuda_amp
        else nullcontext()
    )

    with torch.no_grad():
        with autocast_ctx:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )

    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def main():
    test_json_path = _env("FLICKR8K_TEST_JSON", os.path.join("blip2_lora_finetune", "flickr8k_test.json"))
    image_dir = _env("FLICKR8K_IMAGE_DIR", os.path.join("data", "flickr8k", "Flickr8k_Dataset"))
    output_json_path = _env("INFER_OUTPUT_JSON", "generated_captions.json")
    max_new_tokens = int(_env("INFER_MAX_NEW_TOKENS", "50"))
    num_beams = int(_env("INFER_NUM_BEAMS", "4"))

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    processor, model = load_model_and_processor()

    results = []
    errors = []

    for item in tqdm(test_data, desc="Inference"):
        image_name = item["image"]
        image_path = os.path.join(image_dir, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
            prompt = "Write a caption for this image."
            caption = generate_caption(
                processor=processor,
                model=model,
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            results.append({"image": image_name, "caption": caption})
        except Exception as exc:
            msg = str(exc)
            results.append({"image": image_name, "caption": "[ERROR]", "error": msg})
            errors.append({"image": image_name, "error": msg})

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Saved captions to %s", output_json_path)

    if errors:
        err_path = "inference_errors.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        logger.warning("Saved %d inference errors to %s", len(errors), err_path)


if __name__ == "__main__":
    main()
