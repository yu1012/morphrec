import os
import torch

from glob import glob

from data_loader import build_dataloader
from logger import build_logger
from losses import build_loss_fn
from lr_scheduler import build_scheduler, build_scheduler
from metrics import build_metric
from models.style_unet import StyleUNet
from optimizer import build_optimizer, build_optimizer
import utils

def init_env(cfg):
    torch.set_num_threads(8)
    if cfg.GPUS:
        gpus = ','.join([str(i) for i in cfg.GPUS])
        utils.set_gpu(gpus)
        if len(cfg.GPUS) > 1 and cfg.DDP:
            utils.set_ddp(len(cfg.GPUS))
        cfg.IS_CUDA_AVAILABLE = torch.cuda.is_available()
    else:
        cfg.IS_CUDA_AVAILABLE = False
    
    utils.set_seed(cfg.SEED, cfg.IS_CUDA_AVAILABLE)
    cfg.freeze()

    return cfg

def init_train(cfg):
    log_path = os.path.join(cfg.LOG_PATH, f"{cfg.EXP_ID}")
    utils.make_save_dir(log_path)
    utils.save_cfg(log_path, cfg)

    model = StyleUNet(num_blocks=cfg.NUM_BLOCKS)

    if cfg.IS_CUDA_AVAILABLE:
        model = model.cuda()

    train_dl, val_dl = build_dataloader(cfg, 'train')
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    loss_fn = build_loss_fn(cfg)
    metric = build_metric()
    logger = build_logger(log_path, cfg)

    return model, train_dl, val_dl, optimizer, scheduler, loss_fn, metric, logger, log_path


def init_test(cfg):
    log_path = os.path.join(cfg.LOG_PATH, f"{cfg.EXP_ID}")

    utils.make_save_dir(log_path)
    utils.save_cfg(log_path, cfg)

    model = StyleUNet(num_blocks=cfg.NUM_BLOCKS)

    if cfg.IS_CUDA_AVAILABLE:
        model = model.cuda()

    if cfg.TEST.MODEL_CKPT_PATH:
        print('load model from: ', cfg.TEST.MODEL_CKPT_PATH)
        model.load_state_dict(torch.load(cfg.TEST.MODEL_CKPT_PATH)['model_state_dict'])
    else:
        model_ckpt_flist = sorted(glob(os.path.join(log_path, '*.pth')))
        if len(model_ckpt_flist)>0:
            print('load model from: ', model_ckpt_flist[-1])
            model.load_state_dict(torch.load(model_ckpt_flist[-1])['model_state_dict'])
        else:
            raise ValueError('No Checkpoint to evaluate')
    
    test_dl = build_dataloader(cfg, 'test')
    metric = build_metric()
    loss_fn = build_loss_fn(cfg)

    return model, test_dl, metric, loss_fn, log_path
