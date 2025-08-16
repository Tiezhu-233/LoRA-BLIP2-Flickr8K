# import os
# import json
# import torch
# from tqdm import tqdm
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from peft import PeftModel

# # ==== 配置 ====
# base_model_id = "blip2-opt-2.7b"  # 确保使用正确的模型ID
# lora_checkpoint = "checkpoints/checkpoint-3750"  # 包含 adapter_config.json 和 adapter_model.safetensors
# test_json_path = "blip2_lora_finetune/flickr8k_test.json"
# image_dir = "/root/autodl-tmp/data/Flickr8k/Images"
# output_json_path = "generated_captions.json"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==== 加载模型和 processor ====
# print("🔧 加载模型...")
# processor = Blip2Processor.from_pretrained(base_model_id)

# # 加载基础模型
# base_model = Blip2ForConditionalGeneration.from_pretrained(
#     base_model_id,
#     device_map="auto",
#     load_in_8bit=True,
#     torch_dtype=torch.float16  # 使用半精度减少显存占用
# )

# # 加载LoRA适配器
# try:
#     model = PeftModel.from_pretrained(base_model, lora_checkpoint)
#     print("✅ LoRA适配器加载成功")
#     # 合并LoRA权重到基础模型
#     model = model.merge_and_unload()
#     print("✅ LoRA权重已合并到基础模型")
# except Exception as e:
#     print(f"⚠️ LoRA适配器加载失败: {e}")
#     print("⚠️ 使用基础模型进行推理")
#     model = base_model

# model.eval()
# print("✅ 模型准备就绪")

# # ==== 加载测试集 ====
# print("📂 加载测试集...")
# with open(test_json_path, "r") as f:
#     test_data = json.load(f)

# results = []

# # # ==== 优化后的提示工程 ====
# # def create_prompt(image_path):
# #     """根据图像路径创建更自然的提示"""
# #     filename = os.path.basename(image_path)
# #     return f"Describe the content of the image '{filename}' in a detailed and complete sentence."

# # ==== 遍历测试图像并生成描述 ====
# print("🧠 开始生成描述...")
# for item in tqdm(test_data):
#     image_path = os.path.join(image_dir, item["image"])
    
#     try:
#         image = Image.open(image_path).convert("RGB")
#     except Exception as e:
#         print(f"⚠️ 无法加载图像 {image_path}: {e}")
#         results.append({
#             "image": item["image"],
#             "caption": "[IMAGE LOAD ERROR]",
#             "error": str(e)
#         })
#         continue

#     # 创建更自然的提示
#     prompt = 'Write a caption for this image.'
    
#     try:
#         # 处理输入
#         inputs = processor(
#             images=image, 
#             text=prompt, 
#             return_tensors="pt"
#         ).to(device, torch.float16)
        
#         # 生成描述
#         with torch.no_grad():
#             generated_ids = model.generate(
#                 input_ids=inputs["input_ids"],
#                 pixel_values=inputs["pixel_values"],
#                 attention_mask=inputs["attention_mask"],
#                 max_new_tokens=50,
#                 num_beams=6,  
#                 early_stopping=True,
#                 no_repeat_ngram_size=3,
#                 eos_token_id=model.config.eos_token_id,
#                 )
        
#         # 解码并清理输出
#         caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
#         # 移除提示文本（如果存在）
#         if prompt.lower() in caption.lower():
#             idx = caption.lower().index(prompt.lower())
#             caption = caption[idx + len(prompt):].strip()
        
#         # 清理标点符号
#         caption = caption.strip().strip('"').strip()
#         if caption and caption[0] in ['.', ',', ':', ';']:
#             caption = caption[1:].strip()
        
#         # 确保首字母大写
#         if caption:
#             caption = caption[0].upper() + caption[1:]
        
#     except Exception as e:
#         print(f"⚠️ 生成描述失败: {e}")
#         caption = f"[GENERATION ERROR: {str(e)}]"
    
#     # 实时打印生成的描述
#     print(f"[{item['image']}] => \"{caption}\"")
    
#     results.append({
#         "image": item["image"],
#         "caption": caption
#     })

# # ==== 保存结果 ====
# with open(output_json_path, "w") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)

# print(f"✅ 推理完成，结果已保存至：{output_json_path}")
# print(f"🖼️ 处理图像数量: {len(results)}")

import os
import json
import logging
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, GPT2Tokenizer
from transformers import GPT2TokenizerFast
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer
# ==== 配置日志 ====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==== 配置参数 ====
class Config:
    # 模型配置
    base_model_id = "blip2-opt-2.7b"
    lora_checkpoint = "checkpoint/checkpoint-3750"
    
    # 数据配置
    test_json_path = "blip2_lora_finetune/flickr8k_test.json"
    image_dir = "/root/autodl-tmp/data/Flickr8k/Images"
    output_json_path = "generated_captions.json"
    
    # 推理参数
    max_new_tokens = 100
    num_beams = 5
    temperature = 0.7
    top_p = 0.9
    repetition_penalty = 1.5
    no_repeat_ngram_size = 2
    
    # 系统参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16
    validate_tokenizer = True

config = Config()

# ==== 验证分词器与模型的兼容性 ====
def validate_tokenizer(model, processor):
    """全面验证分词器与模型的兼容性"""
    logger.info("="*50)
    logger.info("开始分词器验证")
    logger.info("="*50)
    
    text_config = model.config.text_config
    tokenizer = processor.tokenizer
    
    # 1. 基本信息检查
    tokenizer_type = type(tokenizer).__name__
    model_lm_type = text_config.model_type
    
    logger.info(f"Tokenizer 类型: {tokenizer_type}")
    logger.info(f"模型语言模型类型: {model_lm_type}")
    logger.info(f"Tokenizer 词汇表大小: {tokenizer.vocab_size}")
    logger.info(f"模型词汇表大小: {text_config.vocab_size}")
    
    # 检查类型匹配
    if tokenizer_type.lower() != model_lm_type.lower():
        logger.warning(f"⚠️ 严重不匹配: Tokenizer类型({tokenizer_type}) != 模型类型({model_lm_type})")
    else:
        logger.info("✅ Tokenizer类型匹配")
    
    # 检查词汇表大小
    if tokenizer.vocab_size != text_config.vocab_size:
        logger.warning(f"⚠️ 严重不匹配: Tokenizer词汇表大小({tokenizer.vocab_size}) != 模型词汇表大小({text_config.vocab_size})")
    else:
        logger.info("✅ 词汇表大小匹配")
    
    # 2. 特殊标记检查
    logger.info("\n特殊标记检查:")
    tokens_to_check = {
        "eos_token": (tokenizer.eos_token_id, text_config.eos_token_id),
        "bos_token": (tokenizer.bos_token_id, text_config.bos_token_id),
        "pad_token": (tokenizer.pad_token_id, text_config.pad_token_id),
        "unk_token": (tokenizer.unk_token_id, getattr(text_config, "unk_token_id", None))
    }
    
    all_match = True
    for name, (tokenizer_id, model_id) in tokens_to_check.items():
        if model_id is None:
            logger.info(f"{name}: 模型未定义")
            continue
            
        if tokenizer_id == model_id:
            logger.info(f"✅ {name}: ID匹配 ({tokenizer_id})")
        else:
            logger.warning(f"⚠️ {name}: 不匹配! Tokenizer={tokenizer_id}, 模型={model_id}")
            all_match = False
    
    # 3. 编码/解码测试
    logger.info("\n编码/解码测试:")
    test_texts = [
        "A cat sitting on a mat",
        "Describe this image in detail:",
        "图像描述：一只狗在公园里奔跑",
        tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>",
        "Special !@#$% characters"
    ]
    
    for text in test_texts:
        try:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded, skip_special_tokens=True)
            
            if decoded == text:
                logger.info(f"✅ '{text}' -> 编码/解码一致")
            else:
                logger.warning(f"⚠️ '{text}' -> 解码为 '{decoded}'")
                
        except Exception as e:
            logger.error(f"❌ 处理 '{text}' 时出错: {str(e)}")
    
    # 4. 嵌入层检查
    try:
        logger.info("\n嵌入层检查:")
        embedding_layer = model.get_input_embeddings()
        logger.info(f"嵌入层大小: {embedding_layer.num_embeddings}")
        
        # 测试边界token
        test_tokens = [0, 1, tokenizer.vocab_size - 1, text_config.vocab_size - 1]
        for token_id in test_tokens:
            try:
                embedding = embedding_layer(torch.tensor([token_id]).to(config.device))
                logger.info(f"Token {token_id} 嵌入成功 (形状: {embedding.shape})")
            except IndexError:
                logger.error(f"❌ Token {token_id} 超出嵌入层范围!")
                
    except Exception as e:
        logger.error(f"无法访问嵌入层: {str(e)}")
    
    logger.info("="*50)
    logger.info("分词器验证完成")
    logger.info("="*50)
    
    return all_match

# ==== 加载模型 ====
def load_model():
    """加载基础模型和LoRA适配器"""
    logger.info("🔧 加载模型...")
    
    # 加载处理器 - 显式指定OPT分词器
    try:
        processor = Blip2Processor.from_pretrained("checkpoint/checkpoint-3750/processor")
        # opt_tokenizer = AutoTokenizer.from_pretrained("opt-2.7b", use_fast=False)
        # processor.tokenizer = opt_tokenizer  # 强制替换为正确 tokenizer
        logger.info(f"✅ 加载处理器完成: {type(processor).__name__}")
    except Exception as e:
        logger.error(f"❌ 加载处理器失败: {str(e)}")
        raise
        
    
    # 加载基础模型
    try:
        base_model = Blip2ForConditionalGeneration.from_pretrained(
            config.base_model_id,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=config.torch_dtype
        )
        logger.info(f"✅ 加载基础模型完成: {base_model.__class__.__name__}")
    except Exception as e:
        logger.error(f"❌ 加载基础模型失败: {str(e)}")
        raise
    
    # 关键修复：手动对齐特殊标记
    base_model.config.text_config.eos_token_id = processor.tokenizer.eos_token_id
    base_model.config.text_config.pad_token_id = processor.tokenizer.pad_token_id
    base_model.config.text_config.bos_token_id = processor.tokenizer.bos_token_id
    
    logger.info(f"✅ 手动对齐特殊标记: "
                f"EOS={base_model.config.text_config.eos_token_id}, "
                f"PAD={base_model.config.text_config.pad_token_id}, "
                f"BOS={base_model.config.text_config.bos_token_id}")
    
    # 加载LoRA适配器
    model = base_model
    if os.path.exists(config.lora_checkpoint):
        try:
            # 检查LoRA配置
            peft_config = PeftConfig.from_pretrained(config.lora_checkpoint)
            logger.info(f"LoRA配置: {peft_config.to_dict()}")
            
            # 加载适配器
            model = PeftModel.from_pretrained(base_model, config.lora_checkpoint)
            logger.info("✅ LoRA适配器加载成功")
            
            # 合并权重
            model = model.merge_and_unload()
            logger.info("✅ LoRA权重已合并到基础模型")
        except Exception as e:
            logger.error(f"⚠️ LoRA适配器加载失败: {str(e)}")
            logger.warning("⚠️ 使用基础模型进行推理")
    else:
        logger.warning(f"⚠️ LoRA检查点不存在: {config.lora_checkpoint}")
        logger.warning("⚠️ 使用基础模型进行推理")
    
    # 验证分词器
    if config.validate_tokenizer:
        logger.info("🔍 验证分词器兼容性...")
        tokenizer_valid = validate_tokenizer(model, processor)
        if not tokenizer_valid:
            logger.warning("⚠️ 分词器验证发现潜在问题，推理结果可能受影响")
    
    model.eval()
    logger.info("✅ 模型准备就绪")
    
    return processor, model

# ==== 创建提示 ====
def create_prompt(image_path):
    """根据图像路径创建提示"""
    filename = os.path.basename(image_path)
    return f"A detailed description of the image '{filename}':"

# ==== 清理生成的描述 ====
def clean_caption(caption, prompt):
    """清理生成的描述文本"""
    # 移除提示文本
    prompt_lower = prompt.lower()
    caption_lower = caption.lower()
    
    if prompt_lower in caption_lower:
        idx = caption_lower.index(prompt_lower)
        caption = caption[idx + len(prompt):].strip()
    
    # 清理开头标点
    while caption and caption[0] in ['.', ',', ':', ';', '-', '—']:
        caption = caption[1:].strip()
    
    # 确保首字母大写
    if caption:
        caption = caption[0].upper() + caption[1:]
    
    # 移除多余的空白
    caption = ' '.join(caption.split())
    logger.info(f"✅ caption: {caption} ")
    return caption

# ==== 主函数 ====
def main():
    # 加载模型和处理器
    processor, model = load_model()
    
    # 加载测试数据
    logger.info("📂 加载测试集...")
    try:
        with open(config.test_json_path, "r") as f:
            test_data = json.load(f)
        logger.info(f"✅ 加载测试集完成: {len(test_data)} 个样本")
    except Exception as e:
        logger.error(f"❌ 加载测试集失败: {str(e)}")
        return
    
    results = []
    error_count = 0
    success_count = 0
    
    # 推理循环
    logger.info("🧠 开始生成描述...")
    for item in tqdm(test_data, desc="生成描述"):
        image_path = os.path.join(config.image_dir, item["image"])
        result_item = {
            "image": item["image"],
            "caption": "",
            "error": None
        }
        
        try:
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            
            # 创建提示
            prompt = create_prompt(item["image"])
            
            # 预处理输入
            inputs = processor(
                images=image, 
                text=prompt, 
                return_tensors="pt"
            ).to(config.device, config.torch_dtype)
            
            # 生成描述 - 关键修复：使用分词器的EOS token
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=config.torch_dtype):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    num_beams=config.num_beams,
                    repetition_penalty=config.repetition_penalty,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    eos_token_id=processor.tokenizer.eos_token_id,  # 关键修复
                    early_stopping=True
                )
            
            # 解码输出
            caption = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # 清理描述
            caption = clean_caption(caption, prompt)
            result_item["caption"] = caption
            success_count += 1
            
            # 实时日志
            if success_count % 50 == 0 or success_count == 1:
                logger.info(f"🖼️ [{item['image']}] => \"{caption}\"")
            
        except Exception as e:
            error_msg = f"处理 {item['image']} 时出错: {str(e)}"
            logger.error(error_msg)
            result_item["error"] = error_msg
            error_count += 1
        
        results.append(result_item)
    
    # 保存结果
    try:
        with open(config.output_json_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 推理完成，结果已保存至: {config.output_json_path}")
    except Exception as e:
        logger.error(f"❌ 保存结果失败: {str(e)}")
    
    # 统计信息
    logger.info(f"📊 总计处理: {len(test_data)} 张图像")
    logger.info(f"✅ 成功: {success_count} ({(success_count/len(test_data))*100:.2f}%)")
    logger.info(f"❌ 失败: {error_count} ({(error_count/len(test_data))*100:.2f}%)")
    
    # 保存错误日志
    if error_count > 0:
        error_log_path = "inference_errors.json"
        errors = [r for r in results if r["error"]]
        with open(error_log_path, "w") as f:
            json.dump(errors, f, indent=2)
        logger.info(f"⚠️ 错误详情已保存至: {error_log_path}")

if __name__ == "__main__":
    main()