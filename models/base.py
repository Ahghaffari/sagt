import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, node_num, input_dim=1, output_dim=1, seq_len=12, horizon=12):
        super().__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

    def param_num(self):
        return sum(p.nelement() for p in self.parameters())
