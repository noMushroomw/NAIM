import math

def cosine_warmup(total_steps, warmup_steps, floor=0.01):

    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(floor, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return fn

def gpt2_lr(step, warmup_steps, max_steps, min_lr_ratio=0.1):

    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr_ratio
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * ratio))
