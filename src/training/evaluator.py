import os

import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm

from src.core.constants import MEANS, STDS
from .metrics import batched_mse, batched_corr

def inference(model, dataloader, loss_fn, cfg, mode):
    model.eval()
    save_all_outputs = mode == 'test'

    output_all = []; target_all = []; fname_all = []
    losses = []; rmse_vals = []; corr_vals = []

    if cfg.PREPROCESSING.GEN_LEAD == ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
        means = torch.Tensor(MEANS[:2]+MEANS[6:]).cuda(non_blocking=True)
        stds = torch.Tensor(STDS[:2]+STDS[6:]).cuda(non_blocking=True)
    elif cfg.PREPROCESSING.GEN_LEAD == ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
        means = torch.Tensor(MEANS[6:]).cuda(non_blocking=True)
        stds = torch.Tensor(STDS[6:]).cuda(non_blocking=True)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_batch = batch['input'].type(torch.FloatTensor)
            target_batch = batch['target'].type(torch.FloatTensor)
            fname_batch = batch['fname']

            if cfg.IS_CUDA_AVAILABLE:
                input_batch = input_batch.cuda(non_blocking=True)
                target_batch = target_batch.cuda(non_blocking=True)

            with torch.amp.autocast('cuda'):
                output_batch = model(input_batch) #, z)
                loss = loss_fn(output_batch, target_batch)
            losses.append(loss.item())

            # rescale output and target if normalization is applied
            if cfg.PREPROCESSING.GLOB_Z_NORM == True:
                output_batch = (output_batch * stds.view(1, -1, 1)) + means.view(1, -1, 1)
                target_batch = (target_batch * stds.view(1, -1, 1)) + means.view(1, -1, 1)
            else:
                output_batch = (output_batch * 5.0)
                target_batch = (target_batch * 5.0)

            rmse_val = batched_mse(output_batch, target_batch).cpu().numpy() ** 0.5
            corr_val = batched_corr(output_batch, target_batch).cpu().numpy()

            rmse_vals.append(rmse_val)
            corr_vals.append(corr_val)

            if save_all_outputs:
                output_all.append(output_batch.detach())
                target_all.append(target_batch.detach())
                fname_all.append(fname_batch)

    rmse_vals = np.concatenate(rmse_vals, axis=0)
    corr_vals = np.concatenate(corr_vals, axis=0)

    if save_all_outputs:
        output_all = torch.cat(output_all, dim=0)
        target_all = torch.cat(target_all, dim=0)
        fname_all = np.concatenate(fname_all, axis=0)

    return np.mean(losses), rmse_vals, corr_vals, output_all, target_all, fname_all


def evaluate(model, loss_fn, dataloader, metrics, cfg, lead_wise_eval=False, save_path=None, mode='val', save_pkl=False):
    loss, mse_vals, corr_vals, reconstructed_signal, gt_signal, fname = inference(model, dataloader, loss_fn, cfg, mode)
    
    summary = {};
    imgs = []
    
    if lead_wise_eval:
        for i, lead in enumerate(cfg.PREPROCESSING.GEN_LEAD):
            summary[f"rmse_{lead}"] = np.mean(mse_vals[:, i])
            summary[f"corr_{lead}"] = np.mean(corr_vals[:, i])
        summary_keys = ", ".join(summary.keys())
        summary_values = ", ".join("{:05.4f}".format(v) for v in summary.values())
        metrics_string = f"{summary_keys}\n{summary_values}\n"
    else:
        summary['rmse'] = np.mean(mse_vals)
        summary['corr'] = np.mean(corr_vals)
        metrics_string = ", ".join("{},{:05.4f}".format(k, v) for k, v in summary.items())
    print(metrics_string)

    if save_pkl:
        print("Saving pkl files...")
        device = reconstructed_signal.device

        raw_12l_signal = [batch['filtered_12l_ecg'] for batch in dataloader]  # List of (N_i, 12, T)
        raw_12l_signal = torch.cat(raw_12l_signal, dim=0).to(device)  # (N, 12, T)

        gt_limb_ecgs = raw_12l_signal[:, :6, :]  # (N, 6, T)

        # denormalize gt_limb_leads
        if cfg.PREPROCESSING.GLOB_Z_NORM:
            limb_means = torch.tensor(MEANS[:6], device=device).view(1, -1, 1)
            limb_stds = torch.tensor(STDS[:6], device=device).view(1, -1, 1)    # (1, 6, 1)
            gt_limb_ecgs = gt_limb_ecgs * limb_stds + limb_means  # (N, 6, T)
        else:
            # If not using lead-wise normalization, scale by 5.0
            gt_limb_ecgs = gt_limb_ecgs * 5.0

        # --- Extract precordial ECGs from reconstructed signal ---
        recon_precordial_ecgs = reconstructed_signal[:, -6:, :]  # (N, 6, T)
        gt_precordial_ecgs = gt_signal[:, -6:, :]  # (N, 6, T)

        recon_ecgs = torch.cat([gt_limb_ecgs, recon_precordial_ecgs], dim=1)  # (N, 12, T)
        gt_ecgs = torch.cat([gt_limb_ecgs, gt_precordial_ecgs], dim=1)  # (N, 12, T)

        reconstructed_signal = reconstructed_signal.detach().cpu().numpy()
        gt_signal = gt_signal.detach().cpu().numpy()
        recon_ecgs = recon_ecgs.detach().cpu().numpy()
        gt_ecgs = gt_ecgs.detach().cpu().numpy()

        save_path_recon_ecg = os.path.join(save_path, 'test_recon_ecg')
        os.makedirs(save_path_recon_ecg, exist_ok=True)
        
        for idx, fn in enumerate(fname):
            path = os.path.join(save_path_recon_ecg, os.path.basename(fn))
            with open(path, 'wb') as f:
                pickle.dump(recon_ecgs[idx], f)

        output_dict = {"fname": fname, "reconstructed_signal": reconstructed_signal, "gt_signal": gt_signal}
        pd.to_pickle(output_dict, os.path.join(save_path, 'outputs.pkl'))

        save_path_gt_ecg = os.path.join(save_path, 'test_gt_ecg')
        os.makedirs(save_path_gt_ecg, exist_ok=True)

        for idx, fn in enumerate(fname):
            path = os.path.join(save_path_gt_ecg, os.path.basename(fn))
            with open(path, 'wb') as f:
                pickle.dump(gt_ecgs[idx], f)
        print("Saving pkl files Done")


    return loss, summary, metrics_string, imgs
