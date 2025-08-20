import torch.optim as optim

def build_scheduler(optimizer, cfg):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode=cfg.LR_SCHEDULER.MODE,
                                                    factor=cfg.LR_SCHEDULER.FACTOR,
                                                    patience=cfg.LR_SCHEDULER.PATIENCE,
                                                    min_lr=cfg.LR_SCHEDULER.MIN_LR,
                                                    verbose=cfg.LR_SCHEDULER.VERBOSE)
    
    return scheduler

