import os
import torch

from tqdm import tqdm

from src.utils.utils import write_log, save_checkpoint
from .evaluator import evaluate


def train_one_epoch(model, optimizer, loss_fn, dataloader, cfg):
    train_loss, num_samples = 0, 0

    model.train()
    optimizer.zero_grad()

    scaler = torch.amp.GradScaler('cuda')

    local_progress=tqdm(dataloader, total=len(dataloader))
    for i, batch in enumerate(local_progress):
        input_batch = batch['input'].type(torch.FloatTensor)
        target_batch = batch['target'].type(torch.FloatTensor)

        if cfg.IS_CUDA_AVAILABLE:
            input_batch = input_batch.cuda(non_blocking=True)
            target_batch = target_batch.cuda(non_blocking=True)

        with torch.amp.autocast('cuda'):
            output_batch = model(input_batch)
            loss = loss_fn(output_batch, target_batch)

        local_progress.set_postfix(loss='{:05.3f}'.format(loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)

        scaler.update()        
        optimizer.zero_grad()

        train_loss += loss.item() * len(input_batch)
        num_samples += len(input_batch)

    train_loss = round(train_loss / num_samples, 6)

    return train_loss


def train_and_evaluate(cfg, model, train_dl, val_dl, optimizer, scheduler, loss_fn, metric, logger, save_path):
    best_val_result = -1e+10
    best_model_path = os.path.join(save_path, 'best_epoch_0.pth')
    early_stopping_cnt = 0
    print(f"Model & Log saved to {save_path}")

    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_dl, cfg)

        val_loss, val_metrics, metrics_string, val_imgs = evaluate(model, loss_fn, val_dl, metric, cfg, False, save_path,
                                                                    mode=f"val_{str(epoch)}")
        val_corr = round(val_metrics['corr'], 6)

        scheduler.step(val_corr)
        print(f'epoch: {str(epoch)} / val_corr: {val_corr:.4f} / val_loss: {val_loss:.4f}')

        # Update Results
        if cfg.LOGGER.USE_LOGGER:
            summary_dict = {"train/loss": train_loss, **{f"metric/{k}": v for k, v in val_metrics.items()}}

            logger.update_scalers(summary_dict)
            logger.update_images(val_imgs)
        write_log(epoch, save_path, 'results_log.txt', metrics_string)
    
    # early stopping
        if val_corr > best_val_result:
            best_val_result = val_corr
            early_stopping_cnt = 0

            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            fname = f'best_epoch_{epoch}_corr_{val_corr}.pth'
            best_model_path = os.path.join(save_path, fname)
            model_state_dict = model.module.state_dict() if len(cfg.GPUS) > 1 else model.state_dict()
            save_checkpoint({'epoch': epoch, 'model_state_dict': model_state_dict},
                                    checkpoint=save_path,
                                    filename=fname)
        else:
            early_stopping_cnt = early_stopping_cnt+1

        if early_stopping_cnt > cfg.TRAIN.EARLY_STOP_PATIENCE:
            print('early stopping')
            break