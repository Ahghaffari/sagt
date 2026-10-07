import torch
import torch.nn as nn

from .testam_core import TESTAMCore


def _masked_mae(preds, labels, null_val=0.0, reduce=True):
    mask = (labels > null_val).float()
    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)
    loss = torch.abs(preds - labels) * mask
    loss = torch.nan_to_num(loss)
    return torch.mean(loss) if reduce else loss


class TESTAM(nn.Module):
    """Adapter: x (B,T,N,C) -> core (B,C,N,T); returns (pred (B,T,N,1), full_loss).
    Training loss = per-expert loss + worst-route-avoidance + best-route-selection."""
    loss_is_aux = True

    def __init__(self, num_nodes, in_dim=2, hidden_size=32, layers=3, dropout=0.3,
                 prob_mul=False, seq_len=12, horizon=12, max_time_index=288,
                 warmup_epochs=0):
        super().__init__()
        self.core = TESTAMCore(num_nodes=num_nodes, dropout=dropout, in_dim=in_dim,
                               out_dim=1, hidden_size=hidden_size, layers=layers,
                               prob_mul=prob_mul, max_time_index=max_time_index)
        self.scaler = None
        self.threshold = 0.0
        self.warmup_epochs = warmup_epochs  # epochs with expert loss only (official warm-start)
        self.cur_epoch = 0

    def set_scaler(self, scaler):
        self.scaler = scaler

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def _get_label(self, ind_loss, gate, real):
        empty_val = (real.permute(0, 2, 3, 1).unsqueeze(-1).expand_as(gate)) <= self.threshold
        max_error = ind_loss.argmax(dim=-1, keepdim=True)
        cur_expert = gate.argmax(dim=-1, keepdim=True)
        incorrect = max_error == cur_expert
        selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0)
        scaling = torch.ones_like(gate) * ind_loss
        scaling = scaling.scatter(-1, max_error, 0.)
        scaling = scaling / (scaling.sum(dim=-1, keepdim=True)) * (1 - selected)
        l_worst = torch.where(incorrect, scaling, selected)
        l_worst = torch.where(empty_val, torch.zeros_like(gate), l_worst).detach()
        min_error = ind_loss.argmin(dim=-1, keepdim=True)
        correct = min_error == cur_expert
        scaling = torch.zeros_like(gate).scatter(-1, min_error, 1.)
        l_best = torch.where(correct, selected, scaling)
        l_best = torch.where(empty_val, torch.zeros_like(gate), l_best).detach()
        return l_worst, l_best

    def forward(self, x, label=None, batches_seen=None, task_level=None):
        inp = x.permute(0, 3, 2, 1)  # (B, C, N, T)
        if self.training:
            out, gate, ind_out = self.core(inp)
            real = self.scaler.inverse_transform(label).permute(0, 3, 2, 1)  # (B,1,N,T)
            ind_loss = _masked_mae(self.scaler.inverse_transform(ind_out),
                                   real.permute(0, 2, 3, 1).unsqueeze(-1),
                                   self.threshold, reduce=False)
            if self.cur_epoch <= self.warmup_epochs:
                loss = ind_loss.mean()
            else:
                l_worst, l_best = self._get_label(ind_loss, gate, real)
                worst = -.5 * l_worst * torch.log(gate)
                best = -.5 * l_best * torch.log(gate)
                loss = ind_loss.mean() + worst.mean() + best.mean()
            pred = out.permute(0, 3, 2, 1)  # (B,1,N,T) -> (B,T,N,1)
            return pred, loss
        out = self.core(inp)
        return out.permute(0, 3, 2, 1)
