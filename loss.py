import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTripletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, labels):
        eps = 1e-7
        dist_mat = torch.pow(feats.unsqueeze(1) - feats.unsqueeze(0), 2).sum(dim=-1).clamp(min=eps).sqrt()

        is_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
        is_neg = labels.unsqueeze(1) != labels.unsqueeze(0)

        dist_ap, dist_an = [], []
        for i in range(len(labels)):
            pos_dists = dist_mat[i][is_pos[i]]
            neg_dists = dist_mat[i][is_neg[i]]
            
            if len(pos_dists) > 0 and len(neg_dists) > 0:
                dist_ap.append(pos_dists.max())
                dist_an.append(neg_dists.min())

        if not dist_ap:
            return torch.tensor(0.0, device=feats.device, requires_grad=True)

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        
        loss = F.softplus(dist_ap - dist_an).mean()
        return loss
