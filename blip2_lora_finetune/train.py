import json
import torch
from transformers import TrainingArguments, Trainer
from blip2_lora_wrapper import load_lora_blip2
from flickr8k_dataset import Flickr8kCaptionDataset
from utils import freeze_blip2_modules, print_trainable_params
import logging

# 配置日志格式和级别
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO  # 或 DEBUG 查看更详细信息
)
logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 将全部输入转到模型所在设备
        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # 自动构造 labels（仅当缺失）
        if "labels" not in inputs and "input_ids" in inputs and "attention_mask" in inputs:
            inputs["labels"] = inputs["input_ids"].clone()
            inputs["labels"][inputs["attention_mask"] == 0] = -100
        
        # 模型前向传播
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            labels=inputs.get("labels")
        )
        
        # 清理
        inputs.pop("inputs_embeds", None)
        loss = outputs.loss
        print(f"Loss: {loss.item():.4f}")

        return (loss, outputs) if return_outputs else loss


# === 加载模型和 processor ===
model, processor = load_lora_blip2()
freeze_blip2_modules(model)
print_trainable_params(model)
model.train()  # 显式启用训练模式

# === 加载 Flickr8k 训练数据 ===
with open("blip2_lora_finetune/flickr8k_train.json", "r") as f:
    annotations = json.load(f)

train_dataset = Flickr8kCaptionDataset(
    annotations=annotations,
    processor=processor,
    image_root="/root/autodl-tmp/data/Flickr8k/Images"
)

# === 训练参数 ===
training_args = TrainingArguments(
    output_dir="./checkpoint",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    learning_rate=5e-5,
    fp16=True,
    save_steps=100,
    save_total_limit=2,
    logging_dir="./logs",
    label_names=["labels"]
)

# === 自定义 collate_fn ===
def collate_fn(batch):
    return {
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "pixel_values": torch.stack([s["pixel_values"] for s in batch]),
        "labels": torch.stack([s["labels"] for s in batch])
    }


# === 初始化 Trainer ===
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.tokenizer,
    data_collator=collate_fn
)

# === 单样本测试前向传播 ===
print("=== 测试单样本前向传播 ===")
test_sample = train_dataset[0]
for k in test_sample:
    test_sample[k] = test_sample[k].unsqueeze(0).to(model.device)

with torch.no_grad():
    test_output = model(
        input_ids=test_sample["input_ids"],
        attention_mask=test_sample["attention_mask"],
        pixel_values=test_sample["pixel_values"],
        labels=test_sample["labels"]
    )
    print(f"测试输出: {test_output}")

# === 开始训练 ===
print("=== 开始训练 ===")
trainer.train()

# === 保存 processor（推荐加在训练完成后）===
print("=== 保存 processor ===")
processor.save_pretrained("./checkpoint/processor")
