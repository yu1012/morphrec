import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PatchAwareReconLoss(nn.Module):
    def __init__(self, lambda_guid=1.0, lambda_corr=0.0, lambda_patch=1.0, patch_size=75):
        super().__init__()
        self.lambda_guid = lambda_guid
        self.lambda_corr = lambda_corr
        self.lambda_patch = lambda_patch
        self.patch_size = patch_size

    def reconstruction_loss(self, generated, target, method="all"):
        if method=="random_lead":
            lead_index = np.random.randint(0, generated.shape[1])
            return F.mse_loss(generated[:, lead_index, :], target[:, lead_index, :])
        else:
            return F.mse_loss(generated, target)
        
    def patchify_and_normalize(self, x, patch_size=75):
        B, C, L = x.shape  # (B, 12, 2250)
        assert L % patch_size == 0, "Sequence length must be divisible by patch size"
        
        num_patches = L // patch_size  # 2250 / 75 = 30
        
        # Step 1: Reshape to get patches
        x_patched = x.view(B, C, num_patches, patch_size)  # (B, 12, 30, 75)

        # Step 2: Normalize each patch independently (along last dim)
        mean = x_patched.mean(dim=-1, keepdim=True)       # (B, 12, 30, 1)
        std = x_patched.std(dim=-1, keepdim=True) + 1e-6  # avoid division by zero
        x_patched_norm = (x_patched - mean) / std         # (B, 12, 30, 75)

        # Step 3: Rearrange to (B, 12 * 30, 75)
        x_out = x_patched_norm.permute(0, 2, 1, 3).reshape(B, C * num_patches, patch_size)
        
        return x_out

    def forward(self, outputs, targets):
        l_rec = self.reconstruction_loss(outputs[:, 2:, :], targets[:, 2:, :], "all")
        loss = l_rec

        if self.lambda_corr > 0:
            l_corr = self.correlation_loss(outputs[:, 2:, :], targets[:, 2:, :])
            loss += self.lambda_corr * l_corr

        if self.lambda_patch > 0:
            patch_loss = self.reconstruction_loss(self.patchify_and_normalize(outputs[:, 2:, :], self.patch_size),
                                                  self.patchify_and_normalize(targets[:, 2:, :], self.patch_size), "all")
            loss += self.lambda_patch * patch_loss

        if self.lambda_guid > 0:
            guidance_loss = self.reconstruction_loss(outputs[:, :2, :], targets[:, :2, :], "all")
            loss += self.lambda_guid * guidance_loss
        
        return loss


def build_loss_fn(cfg):
    loss_name = cfg.LOSS.NAME

    if loss_name == 'mae':
        return F.l1_loss
    elif loss_name == 'mse':
        return F.mse_loss
    elif loss_name == 'parl':
        return PatchAwareReconLoss(lambda_guid=cfg.LOSS.LAMBDA_GUID,
                                   lambda_corr=0.0,
                                   lambda_patch=cfg.LOSS.LAMBDA_PATCH,
                                   patch_size=cfg.LOSS.PATCH_SIZE)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

