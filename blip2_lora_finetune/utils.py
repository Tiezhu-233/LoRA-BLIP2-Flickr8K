def freeze_blip2_modules(model):
    for n, p in model.vision_model.named_parameters():
        p.requires_grad = False
    for n, p in model.qformer.named_parameters():
        if not any(x in n for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            p.requires_grad = False
    for n, p in model.language_model.named_parameters():
        if not any(x in n for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            p.requires_grad = False

def print_trainable_params(model):
    print("Trainable Parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"✅ {n}")
