def _keep_trainable(name: str) -> bool:
    if "lora_" in name:
        return True
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value", "q", "k", "v", "o")
    return any(t in name for t in targets)


def freeze_blip2_modules(model):
    for _, p in model.vision_model.named_parameters():
        p.requires_grad = False

    for n, p in model.qformer.named_parameters():
        p.requires_grad = _keep_trainable(n)

    for n, p in model.language_model.named_parameters():
        p.requires_grad = _keep_trainable(n)


def print_trainable_params(model):
    print("Trainable Parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f" {n}")
