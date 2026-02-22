import json
import torch
import os
from transformers import TrainingArguments, Trainer
from blip2_lora_loader import load_lora_blip2
from flickr8k_train_dataset import Flickr8kCaptionDataset
from model_train_utils import freeze_blip2_modules, print_trainable_params
import logging
import inspect

# Comment translated to English and cleaned.
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO  # Comment translated to English and cleaned.
)
logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Comment translated to English and cleaned.
        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Comment translated to English and cleaned.
        if "labels" not in inputs and "input_ids" in inputs and "attention_mask" in inputs:
            inputs["labels"] = inputs["input_ids"].clone()
            inputs["labels"][inputs["attention_mask"] == 0] = -100
        
        # Comment translated to English and cleaned.
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            labels=inputs.get("labels")
        )
        
        # Comment translated to English and cleaned.
        inputs.pop("inputs_embeds", None)
        loss = outputs.loss
        print(f"Loss: {loss.item():.4f}")

        return (loss, outputs) if return_outputs else loss


# Comment translated to English and cleaned.
model, processor = load_lora_blip2()
freeze_blip2_modules(model)
print_trainable_params(model)
model.train()  # Comment translated to English and cleaned.

# Comment translated to English and cleaned.
train_json_path = os.getenv(
    "FLICKR8K_TRAIN_JSON", os.path.join("blip2_lora_finetune", "flickr8k_train.json")
)
image_root = os.getenv("FLICKR8K_IMAGE_DIR", os.path.join("data", "flickr8k", "Flickr8k_Dataset"))

with open(train_json_path, "r") as f:
    annotations = json.load(f)

train_dataset = Flickr8kCaptionDataset(
    annotations=annotations,
    processor=processor,
    image_root=image_root
)

# Comment translated to English and cleaned.
training_args = TrainingArguments(
    output_dir=os.getenv("TRAIN_OUTPUT_DIR", "./checkpoint"),
    per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "8")),
    num_train_epochs=float(os.getenv("TRAIN_NUM_EPOCHS", "5")),
    max_steps=int(os.getenv("TRAIN_MAX_STEPS", "-1")),
    learning_rate=float(os.getenv("TRAIN_LR", "5e-5")),
    fp16=torch.cuda.is_available() and os.getenv("TRAIN_FP16", "1") == "1",
    save_steps=int(os.getenv("TRAIN_SAVE_STEPS", "100")),
    save_total_limit=int(os.getenv("TRAIN_SAVE_TOTAL_LIMIT", "2")),
    logging_dir=os.getenv("TRAIN_LOG_DIR", "./logs"),
    label_names=["labels"]
)

# Comment translated to English and cleaned.
def collate_fn(batch):
    return {
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "pixel_values": torch.stack([s["pixel_values"] for s in batch]),
        "labels": torch.stack([s["labels"] for s in batch])
    }


# Comment translated to English and cleaned.
trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "data_collator": collate_fn,
}
trainer_init_params = inspect.signature(Trainer.__init__).parameters
if "tokenizer" in trainer_init_params:
    trainer_kwargs["tokenizer"] = processor.tokenizer
elif "processing_class" in trainer_init_params:
    trainer_kwargs["processing_class"] = processor

trainer = CustomTrainer(**trainer_kwargs)

# Comment translated to English and cleaned.
print("===  ===")
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
    print(f": {test_output}")

# Comment translated to English and cleaned.
print("===  ===")
trainer.train()

# Comment translated to English and cleaned.
print("===  processor ===")
processor.save_pretrained("./checkpoint/processor")
