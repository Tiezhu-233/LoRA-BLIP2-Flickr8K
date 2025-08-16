from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoTokenizer, BlipImageProcessor
from peft import get_peft_model, prepare_model_for_kbit_training
from lora_config import get_lora_config
import logging
import os

logger = logging.getLogger(__name__)

def load_lora_blip2(
    model_id="./blip2-opt-2.7b",
    # tokenizer_path="./opt-2.7b",  # ✅ 本地 tokenizer 路径
    image_processor_path="./blip2-opt-2.7b"  # ✅ 本地 image_processor 路径
):

    # === 本地加载分词器 & 图像处理器 ===
    # assert os.path.exists(tokenizer_path), "Tokenizer path does not exist!"
    # assert os.path.exists(image_processor_path), "Image processor path does not exist!"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    logger.info(f"✅ 加载 tokenizer: {type(tokenizer).__name__}")
    logger.info(f"✅ 分词器词表大小: {len(tokenizer)}")
    logger.info(f"✅ 特殊标记 ID - EOS: {tokenizer.eos_token_id}, BOS: {tokenizer.bos_token_id}, PAD: {tokenizer.pad_token_id}")
    image_processor = BlipImageProcessor.from_pretrained(image_processor_path,local_files_only=True)
    
    logger.info(f"✅ 加载 image_processor: {type(image_processor).__name__}")
    logger.info(f"✅ 图像尺寸参数: {image_processor.size}, do_resize={image_processor.do_resize}, do_normalize={image_processor.do_normalize}")
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer,local_files_only=True)

    # 日志打印
    logger.info(f"✅ 加载 tokenizer: {type(processor.tokenizer).__name__}")
    logger.info(f"✅ 分词器词表大小: {len(processor.tokenizer)}")
    logger.info(f"✅ 特殊标记 ID - EOS: {processor.tokenizer.eos_token_id}, BOS: {processor.tokenizer.bos_token_id}, PAD: {processor.tokenizer.pad_token_id}")

    # 加载 BLIP-2 模型本体（假设支持从 huggingface 或本地）
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=True
    )

    # LoRA 准备
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())

    return model, processor
