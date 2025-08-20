import os
import pickle as pkl
import random
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import datetime

def set_gpu(gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

def set_ddp(world_size):
    """Set up Distributed Data Parallel environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = '0'

def set_seed(seed, is_cuda):
    # Set the random seed for reproducible experiments
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    if is_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)

def load_yaml(filename):
    with open(filename) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pkl.load(f)
    return ret_di

def make_save_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(os.path.join(directory, "val"))
        os.makedirs(os.path.join(directory, "test"))

def save_checkpoint(state, checkpoint, filename):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    torch.save(state, filepath)

def save_cfg(log_path, cfg):
    from contextlib import redirect_stdout
    with open(log_path + '/yaml_config.yaml', 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

def save_yaml(path, params):
    with open(path, 'w') as file:
        yaml.dump(params, file)

# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_classification.py#L1287
def prf_divide(numerator, denominator, zero_divide_return=0.):
    mask = denominator == 0.
    if not isinstance(mask, Iterable):
        result = np.ones_like(numerator) * zero_divide_return if mask else numerator / denominator
        return result
    denominator = denominator.copy()
    denominator[mask] = 1
    result = numerator / denominator
    if not np.any(mask):
        return result
    result[mask] = zero_divide_return
    return result

def write_log(epoch, path, name, data_):
    data_ = 'epoch_' +str(epoch)+', ' + data_
    if epoch==0:
        with open(path+ '/' + name, 'w') as f:
            f.write(data_)
            f.close()
    else:
        with open(path+ '/' + name, 'a+') as f:
            f.write('\n' + data_)
            f.close()

def viz_gen_ecg(gt_signal, gen_signal, gen_leads, save_path, column_num=4, row_num=5, figsize_x=16, figsize_y=15, fname=None):
    fig, axs = plt.subplots(row_num, column_num, figsize=(figsize_x, figsize_y))
    fig.tight_layout(h_pad=2, w_pad=2)

    if column_num == 1:
        axs[0].set_title(f'{fname}', y=1.0, pad=14)
    elif column_num == 2:
        axs[0, 0].set_title('Groundtruth', y=1.0, pad=14)
        axs[0, 1].set_title('Reconstructed', y=1.0, pad=14)

    for i in range(row_num):
        gt_lead_info = gt_signal[i]
        gen_lead_info = gen_signal[i]

        ylim_max = max(gt_lead_info.max(), gen_lead_info.max())
        ylim_min = min(gt_lead_info.min(), gen_lead_info.min())
        
        if column_num == 1:
            # gt ecg viz
            axs[i].plot(gt_lead_info, linestyle="-", linewidth="0.7", color="blue", label="gt")
            axs[i].tick_params(labelbottom=False, labelleft=True)
            axs[i].set_ylim([ylim_min, ylim_max])
            # gen ecg viz
            axs[i].plot(gen_lead_info, linestyle="-", linewidth="0.7", color="red", label="pred")
            axs[i].tick_params(labelbottom=False, labelleft=True)
            axs[i].set_ylim([ylim_min, ylim_max])
            
            axs[i].set_ylabel(gen_leads[i], rotation=0, labelpad=20)

        elif column_num == 2:
            # gt ecg viz
            axs[i, 0].plot(gt_lead_info, linestyle="-", linewidth="0.7", color="blue", label="gt")
            axs[i, 0].tick_params(labelbottom=False, labelleft=True)
            axs[i, 0].set_ylim([ylim_min, ylim_max])
            # gen ecg viz
            axs[i, 1].plot(gen_lead_info, linestyle="-", linewidth="0.7", color="red", label="pred")
            axs[i, 1].tick_params(labelbottom=False, labelleft=True)
            axs[i, 1].set_ylim([ylim_min, ylim_max])

            axs[i, 0].set_ylabel(gen_leads[i], rotation=0, labelpad=20)

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba()) # convert img to show in tensorboard
    img = torch.from_numpy(np.transpose(img.astype("float32") / 255, (2,0,1))) # convert img to show in tensorboard
    
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return img
