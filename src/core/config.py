from yacs.config import CfgNode as CN
from typing import Optional

_CFG = CN()

_CFG.NUM_BLOCKS: Optional[int] = None
_CFG.EXP_ID: Optional[str] = None
_CFG.TRAIN_TEST_MODE: Optional[str] = None
_CFG.GPUS: Optional[list[int]] = None
_CFG.SEED: Optional[int] = None
_CFG.LOG_PATH: Optional[str] = None

_CFG.DATASET = CN()
_CFG.DATASET.WAVEFORM_PATH: Optional[str] = None
_CFG.DATASET.INDEX_PATH: Optional[str] = None
_CFG.DATASET.TRAIN_INDEX_FNAME: Optional[str] = None
_CFG.DATASET.VAL_INDEX_FNAME: Optional[list[str]] = None
_CFG.DATASET.TEST_INDEX_FNAME: Optional[str] = None
_CFG.DATASET.FILENAME_COLUMN: Optional[str] = None

_CFG.LOGGER = CN()
_CFG.LOGGER.USE_LOGGER: Optional[bool] = None
_CFG.LOGGER.USE_MATPLOTLIB: Optional[bool] = None
_CFG.LOGGER.USE_TENSORBOARD: Optional[bool] = None

_CFG.LOSS = CN()
_CFG.LOSS.NAME: Optional[str] = None

_CFG.LOSS.LAMBDA_GUID: Optional[float] = None
_CFG.LOSS.LAMBDA_PATCH: Optional[float] = None
_CFG.LOSS.PATCH_SIZE: Optional[int] = None

_CFG.OPTIMIZER = CN()
_CFG.OPTIMIZER.NAME: Optional[str] = None
_CFG.OPTIMIZER.LR: Optional[float] = None
_CFG.OPTIMIZER.WEIGHT_DECAY: Optional[float] = None

_CFG.LR_SCHEDULER = CN()
_CFG.LR_SCHEDULER.FACTOR: Optional[float] = None
_CFG.LR_SCHEDULER.MIN_LR: Optional[float] = None
_CFG.LR_SCHEDULER.MODE: Optional[str] = None
_CFG.LR_SCHEDULER.PATIENCE: Optional[int] = None
_CFG.LR_SCHEDULER.VERBOSE: Optional[bool] = None

_CFG.DATALOADER = CN()
_CFG.DATALOADER.BATCH_SIZE: Optional[int] = None

_CFG.TRAIN = CN()
_CFG.TRAIN.NUM_EPOCHS: Optional[int] = None
_CFG.TRAIN.VAL_METRIC: Optional[str] = None
_CFG.TRAIN.EARLY_STOP_PATIENCE: Optional[int] = None
_CFG.TRAIN.TASK: Optional[str] = None
_CFG.TRAIN.VAL_VIZ_N_SAMPLE: Optional[int] = None

_CFG.TEST = CN()
_CFG.TEST.MODEL_CKPT_PATH: Optional[str] = None

_CFG.PREPROCESSING = CN()
_CFG.PREPROCESSING.COND_LEAD: Optional[list[str]] = None
_CFG.PREPROCESSING.GEN_LEAD: Optional[list[str]] = None
_CFG.PREPROCESSING.GLOB_Z_NORM: Optional[bool] = None
_CFG.PREPROCESSING.SAMPLE_RATE: Optional[int] = None
_CFG.PREPROCESSING.SIGNAL_LEN_CUT_SEC: Optional[float] = None
_CFG.PREPROCESSING.NUM_CROPS: Optional[int] = None
_CFG.PREPROCESSING.HIGH_PASS_FILTER = CN()
_CFG.PREPROCESSING.HIGH_PASS_FILTER.IS_ENABLED: Optional[bool] = None
_CFG.PREPROCESSING.HIGH_PASS_FILTER.CUT_OFF: Optional[float] = None
_CFG.PREPROCESSING.HIGH_PASS_FILTER.ORDER: Optional[int] = None
_CFG.PREPROCESSING.LOW_PASS_FILTER = CN()
_CFG.PREPROCESSING.LOW_PASS_FILTER.IS_ENABLED: Optional[bool] = None
_CFG.PREPROCESSING.LOW_PASS_FILTER.CUT_OFF: Optional[float] = None
_CFG.PREPROCESSING.LOW_PASS_FILTER.ORDER: Optional[int] = None

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _CFG.clone()

def parse(cfg_file_path: str):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file_path)
    return cfg
