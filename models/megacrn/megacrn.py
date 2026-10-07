import torch
import torch.nn as nn

from .megacrn_core import MegaCRNCore


def _masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= mask.mean()
    mask = torch.nan_to_num(mask)
    loss = torch.abs(preds - labels) * mask
    return torch.nan_to_num(loss).mean()


class MegaCRN(nn.Module):
    loss_is_aux = True

    def __init__(self, num_nodes, input_dim=1, output_dim=1, horizon=12, rnn_units=64,
                 num_layers=1, cheb_k=3, mem_num=20, mem_dim=64, cl_decay_steps=2000,
                 lamb=0.01, lamb1=0.01, ycov_dim=1):
        super().__init__()
        self.core = MegaCRNCore(num_nodes=num_nodes, input_dim=input_dim, output_dim=output_dim,
                                horizon=horizon, rnn_units=rnn_units, num_layers=num_layers,
                                cheb_k=cheb_k, ycov_dim=ycov_dim, mem_num=mem_num, mem_dim=mem_dim,
                                cl_decay_steps=cl_decay_steps, use_curriculum_learning=True)
        self.input_dim = input_dim
        self.lamb, self.lamb1 = lamb, lamb1
        self.scaler = None
        self.sep_loss = nn.TripletMarginLoss(margin=1.0)
        self.compact_loss = nn.MSELoss()

    def set_scaler(self, scaler):
        self.scaler = scaler

    def _future_tod(self, x):
        H = self.core.horizon
        if x.shape[-1] <= 1:
            return torch.zeros(x.shape[0], H, x.shape[2], 1, device=x.device)
        tod = x[..., 1]
        delta = (tod[:, -1] - tod[:, -2]) % 1.0
        last = tod[:, -1]
        steps = torch.arange(1, H + 1, device=x.device).view(1, H, 1)
        fut = (last.unsqueeze(1) + steps * delta.unsqueeze(1)) % 1.0
        return fut.unsqueeze(-1)

    def forward(self, x, label=None, batches_seen=None, task_level=None):
        xin = x[..., :self.input_dim]
        ycov = self._future_tod(x)
        labels_norm = label[..., :1] if label is not None else None
        out, h_att, query, pos, neg = self.core(xin, ycov, labels_norm, batches_seen)
        if self.training:
            pred = self.scaler.inverse_transform(out)
            true = label[..., :1] if label is not None else out
            true = self.scaler.inverse_transform(true)
            loss1 = _masked_mae(pred, true)
            loss2 = self.sep_loss(query, pos.detach(), neg.detach())
            loss3 = self.compact_loss(query, pos.detach())
            return out, loss1 + self.lamb * loss2 + self.lamb1 * loss3
        return out
