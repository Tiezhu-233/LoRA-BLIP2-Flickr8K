#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import os
import torch
import re
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def generate_image_caption(image, processor, model, device):

    # caption_prompt = ('Identify which objects are present in the diagram and specify how they are related. Based on these findings, please generate a description for this image.')
    caption_prompt = ('Write a caption for this image.')
    
 
    inputs = processor(images=image, text=caption_prompt, return_tensors="pt").to(device)
    # outputs = model.generate(
    #         input_ids=inputs.input_ids,
    #         pixel_values=inputs.pixel_values,
    #         attention_mask=inputs.attention_mask,
    #         do_sample=True,
    #         max_new_tokens=60,
    #         num_return_sequences=3,
    #         temperature=0.8,
    #         top_k=50,
    #         top_p=0.95,
    #         repetition_penalty=1.5,     
    #         no_repeat_ngram_size=3,
    #         early_stopping=True,
    #         eos_token_id=model.config.eos_token_id,

    #     )
    
   
    
    # outputs = model.generate(
    #         input_ids=inputs.input_ids,
    #         pixel_values=inputs.pixel_values,
    #         num_return_sequences=3,  # 每个样本返回3个候选序列
    #         attention_mask=inputs.attention_mask,
    #         max_new_tokens=80,
    #         num_beams=6,
    #         num_beam_groups=3,    
    #         temperature=0.7, 
    #         early_stopping=True,
    #         repetition_penalty=2.0,
    #         diversity_penalty=1.0,
    #         no_repeat_ngram_size=3,
    #         eos_token_id=model.config.eos_token_id,
    #     )
    outputs = model.generate(
             input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            num_beams=3,  
            early_stopping=True,
            no_repeat_ngram_size=3,
            eos_token_id=model.config.eos_token_id,
        )
    print(f"generate 返回张量 shape: {outputs.shape}")
    
    # 处理生成结果
    raw_caption = processor.decode(outputs[0], skip_special_tokens=True)
    for i, cap in enumerate(outputs):
        print(f"[Caption {i+1}]: {processor.decode(cap, skip_special_tokens=True)}")
    print(f"\n[Raw output]: {raw_caption}")
    
    # 后处理函数
    def clean_caption(text, prompt):
        # 移除提示词部分
        text = re.sub(f"^{re.escape(prompt)}", "", text).strip()
        # 清理编号和多余符号
        text = re.sub(r'^\s*\d+[\.\:]?\s*', '', text)  # 匹配1. 或1: 等格式
        text = re.sub(r'[.;,]+$', '', text)            # 移除尾部标点
        return text
        # # 移除重复内容（处理模型输出重复问题）
        # sentences = text.split('.')
        # unique_sentences = []
        # for sent in sentences:
        #     sent = sent.strip()
        #     if sent and sent not in unique_sentences:
        #         unique_sentences.append(sent)
        # return '. '.join(unique_sentences).strip()
    
    # 清理原始描述
    cleaned = clean_caption(raw_caption, caption_prompt)
    
    # 确保长度在15-30词之间
    # words = raw_caption.split()
    # if len(words) > 100:
    #     cleaned = ' '.join(words[:50])
    # elif len(words) < 15:
    #     # 如果太短，使用原始描述（不包含提示词）
    #     cleaned = clean_caption(raw_caption, caption_prompt)
    
    
    return cleaned


def main():
    # 设置全局随机种子保证可复现性
    torch.manual_seed(20250723)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20250723)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 动态精度设置（兼容不同设备）
    if device == "cuda" and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    try:
        
        model_name = "blip2-opt-2.7b"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(device)
        model.eval()
        print(f"Model '{model_name}' loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 使用安全的路径处理
    img_path = "/root/autodl-tmp/data/Flickr8k/Images/1000268201_693b08cb0e.jpg"
    img_path = os.path.expanduser(img_path)  # 处理~扩展
    
    if not os.path.isfile(img_path):
        print(f"Error: Cannot find image at {img_path}")
        return
    
    try:
        image = Image.open(img_path).convert("RGB")
        print(f"\nProcessing image: {os.path.basename(img_path)}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    try:
        caption = generate_image_caption(image, processor, model, device)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory, trying with CPU...")
            device = "cpu"
            model = model.to(device)
            caption = generate_image_caption(image, processor, model, device)
        else:
            print(f"Generation error: {e}")
            return

    # word_count = len(caption[0].split())
    # print(f"\nFinal Caption ({word_count} words):")
    print(f"\nFinal Caption : {caption}")


if __name__ == "__main__":
    main()