"""
ECG Signal Preprocessing Module

- Cropping
- Resampling
- Filtering
- Standardization
"""

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

from src.utils.utils import prf_divide
from src.core.constants import MEANS, STDS


class ECGProcessing:
    def __init__(
        self,
        mode: str,
        sample_rate: int,
        use_high_pass_filter: bool,
        high_pass_filter_cut_off: float,
        high_pass_filter_order: int,
        use_low_pass_filter: bool,
        low_pass_filter_cut_off: float,
        low_pass_filter_order: int,
        signal_len_cut_sec: float,
        num_crops: int = 1,
        **kwargs
    ):
        self.training = mode == 'train'
        self.crop_signal_time_length = signal_len_cut_sec
        self.num_crops = num_crops
        self.resampled_sample_rate = sample_rate
        self.use_filter = any([use_high_pass_filter, use_low_pass_filter])
        self.filters = []
        if use_high_pass_filter:
            self.filters.append(butter(high_pass_filter_order, high_pass_filter_cut_off, btype='highpass',
                                       fs=sample_rate, output='sos'))
        if use_low_pass_filter:
            self.filters.append(butter(low_pass_filter_order, low_pass_filter_cut_off, btype='lowpass',
                                       fs=sample_rate, output='sos'))
        self.glob_z_norm = kwargs.get('glob_z_norm', False)
        
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_crop_ranges(self, signal, sample_rate):
        signal_length = signal.shape[-1]
        crop_length = int(self.crop_signal_time_length * sample_rate)
        assert signal_length >= crop_length, \
            f'{signal_length} is shorter than {crop_length}. set [signal_len_cut_sec] properly in yaml.'
        if self.training:  # random cropping
            i = int(np.random.uniform(low=0, high=signal_length - crop_length, size=1))
            return [[i, i + crop_length]]
        else:
            if self.num_crops == 1:
                return [[0, crop_length]]
            start_idx = np.arange(0,
                                  signal_length - crop_length + 1,
                                  (signal_length - crop_length) // (self.num_crops - 1))
            return [[start, start + crop_length] for start in start_idx]

    def cropping(self, signal, sample_rate):
        crop_ranges = self.get_crop_ranges(signal, sample_rate)
        samples = []
        for start, end in crop_ranges:
            samples.append(signal[:, start:end])
        return np.stack(samples)

    def resampling(self, signal, original_sample_rate):
        target_length = int(signal.shape[-1] * self.resampled_sample_rate / original_sample_rate)
        resampled_signal = resample(signal, target_length, axis=-1)
        return resampled_signal

    def filtering(self, signal):
        filtered_signal = signal
        for sos_filter in self.filters:
            filtered_signal = sosfiltfilt(sos_filter, filtered_signal)
        return filtered_signal

    def normalization(self, signal):
        outputs = []
        max_scale = 5.0
        for sig_sample in signal:
            if self.glob_z_norm:
                lead_wise_outputs = []
                for i, lead_sample in enumerate(sig_sample):
                    m = MEANS[i]
                    s = STDS[i]
                    lead_sample = prf_divide(lead_sample - m, s)
                    lead_wise_outputs.append(lead_sample)
                outputs.append(np.stack(lead_wise_outputs))
            else:
                outputs.append(sig_sample/max_scale)
        outputs = np.stack(outputs)
        return outputs

    def __call__(self,
                 signal: np.ndarray,
                 original_sample_rate: int):
        """
        Cropping -> (Resampling) -> (Filtering) -> Standardization
        """
        samples = self.cropping(signal, original_sample_rate)

        if original_sample_rate != self.resampled_sample_rate:
            samples = self.resampling(samples, original_sample_rate)

        if self.use_filter:
            samples = self.filtering(samples)

        samples = self.normalization(samples)

        return samples.squeeze(0)

    @classmethod
    def init_from_params(cls, mode, cfg):
        instance = cls(mode=mode,
                       sample_rate=cfg.PREPROCESSING.SAMPLE_RATE,
                       use_high_pass_filter=cfg.PREPROCESSING.HIGH_PASS_FILTER.IS_ENABLED,
                       high_pass_filter_cut_off=cfg.PREPROCESSING.HIGH_PASS_FILTER.CUT_OFF,
                       high_pass_filter_order=cfg.PREPROCESSING.HIGH_PASS_FILTER.ORDER,
                       use_low_pass_filter=cfg.PREPROCESSING.LOW_PASS_FILTER.IS_ENABLED,
                       low_pass_filter_cut_off=cfg.PREPROCESSING.LOW_PASS_FILTER.CUT_OFF,
                       low_pass_filter_order=cfg.PREPROCESSING.LOW_PASS_FILTER.ORDER,
                       signal_len_cut_sec=cfg.PREPROCESSING.SIGNAL_LEN_CUT_SEC,
                       num_crops=cfg.PREPROCESSING.NUM_CROPS,
                       glob_z_norm=cfg.PREPROCESSING.GLOB_Z_NORM)
        return instance
