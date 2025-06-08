import os
from typing import Literal

import numpy as np
from torch.utils.data import Dataset

from preprocessing import ECGProcessing
from utils import load_dict


class ECGDataset(Dataset):
    _LEAD_KEYS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def __init__(self,
                 path_x: str,
                 df,
                 cfg,
                 mode: Literal['train', 'valid', 'test']):
        
        print(f'{mode} mode data size is {df.values.shape}')

        self.df = df
        self.filenames = df[cfg.DATASET.FILENAME_COLUMN].tolist()
        self.filepaths = [os.path.join(path_x, f) for f in self.filenames]
        self.cond_lead_index = [self._LEAD_KEYS.index(lead) for lead in cfg.PREPROCESSING.COND_LEAD]
        self.gen_lead_index = [self._LEAD_KEYS.index(lead) for lead in cfg.PREPROCESSING.GEN_LEAD]
        self.mode = mode
        self.cfg = cfg
        self.sample_rates = self.df['SAMPLE_RATE'].values
        self.signal_processor = ECGProcessing.init_from_params(mode, cfg)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx) -> dict:  # get one data point
        sample_rate = int(self.sample_rates[idx])
        raw_ecg = load_dict(self.filepaths[idx])

        if isinstance(raw_ecg, dict):
            ecg = np.stack([raw_ecg[lead] for lead in self._LEAD_KEYS])
        elif isinstance(raw_ecg, np.ndarray):
            ecg = raw_ecg[[self._LEAD_KEYS.index(lead) for lead in self._LEAD_KEYS]]

        ecg = self.signal_processor(ecg, original_sample_rate=sample_rate)

        cond_ecg = ecg[self.cond_lead_index, :]
        gen_ecg = ecg[self.gen_lead_index, :]

        data = {'input': cond_ecg.copy(), 'target': gen_ecg.copy(), 'fname':self.filenames[idx],
                'filtered_12l_ecg': ecg.copy(), 'raw_12l_ecg': raw_ecg.copy()}
        return data
