import os
import pandas as pd
from torch.utils.data import DataLoader

from dataset import ECGDataset


def fetch_dataloader(mode, df, cfg):
    waveform_path = cfg.DATASET.WAVEFORM_PATH
    dataset = ECGDataset(waveform_path, df, cfg, mode)

    if mode == 'train':
        dl = DataLoader(dataset,
                        batch_size=cfg.DATALOADER.BATCH_SIZE,
                        num_workers=8,
                        persistent_workers=True,
                        prefetch_factor=10,
                        pin_memory=cfg.IS_CUDA_AVAILABLE,
                        shuffle=True)
    else:
        dl = DataLoader(dataset,
                        batch_size=cfg.DATALOADER.BATCH_SIZE,
                        num_workers=8,
                        persistent_workers=True,
                        prefetch_factor=10,
                        shuffle=False)
    return dl


def build_dataloader(cfg, mode):
    if mode == 'train':
        train_index = pd.read_pickle(os.path.join(cfg.DATASET.INDEX_PATH, cfg.DATASET.TRAIN_INDEX_FNAME))
        train_dl = fetch_dataloader('train', train_index, cfg)

        val_index = pd.read_pickle(os.path.join(cfg.DATASET.INDEX_PATH, cfg.DATASET.VAL_INDEX_FNAME))
        val_dl = fetch_dataloader('valid', val_index, cfg)

        return train_dl, val_dl
    
    elif mode == 'test':
        test_index = pd.read_pickle(os.path.join(cfg.DATASET.INDEX_PATH, cfg.DATASET.TEST_INDEX_FNAME))
        test_dl = fetch_dataloader('test', test_index, cfg)

        return test_dl
    