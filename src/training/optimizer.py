import torch.optim as optim

def build_optimizer(model, cfg):
    if cfg.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LR, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    elif cfg.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.OPTIMIZER.LR, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.OPTIMIZER.NAME}")

    return optimizer
