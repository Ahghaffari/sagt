import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from common.graph_utils import normalize_adj


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.conv = nn.Conv2d(c_in, c_out, (1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv(x)
        if self.c_in < self.c_out:
            b, _, t, n = x.shape
            pad = torch.zeros(b, self.c_out - self.c_in, t, n, device=x.device)
            return torch.cat([x, pad], dim=1)
        return x


class CausalConv2d(nn.Conv2d):
    def __init__(self, c_in, c_out, kernel_size, dilation=1):
        kernel_size = nn.modules.utils._pair(kernel_size)
        dilation = nn.modules.utils._pair(dilation)
        super().__init__(c_in, c_out, kernel_size, stride=1, padding=0, dilation=dilation)


class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out):
        super().__init__()
        self.Kt, self.c_out = Kt, c_out
        self.align = Align(c_in, c_out)
        self.causal_conv = CausalConv2d(c_in, 2 * c_out, (Kt, 1))

    def forward(self, x):
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        xc = self.causal_conv(x)
        p = xc[:, :self.c_out]
        q = xc[:, -self.c_out:]
        return torch.mul((p + x_in), torch.sigmoid(q))


class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso):
        super().__init__()
        self.Ks = Ks
        self.register_buffer("gso", gso)
        self.weight = nn.Parameter(torch.empty(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.empty(c_out))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        if self.Ks == 1:
            xs = [x]
        else:
            x1 = torch.einsum("hi,btij->bthj", self.gso, x)
            xs = [x, x1]
            for k in range(2, self.Ks):
                xs.append(torch.einsum("hi,btij->bthj", 2 * self.gso, xs[-1]) - xs[-2])
        x = torch.stack(xs, dim=2)
        out = torch.einsum("btkhi,kij->bthj", x, self.weight) + self.bias
        return out


class GraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso):
        super().__init__()
        self.align = Align(c_in, c_out)
        self.cheb = ChebGraphConv(c_out, c_out, Ks, gso)

    def forward(self, x):
        x_in = self.align(x)
        x_gc = self.cheb(x_in).permute(0, 3, 1, 2)
        return x_gc + x_in


class STConvBlock(nn.Module):
    def __init__(self, Kt, Ks, node_num, c_in, channels, gso, dropout):
        super().__init__()
        self.tmp1 = TemporalConvLayer(Kt, c_in, channels[0])
        self.graph = GraphConvLayer(channels[0], channels[1], Ks, gso)
        self.tmp2 = TemporalConvLayer(Kt, channels[1], channels[2])
        self.ln = nn.LayerNorm([node_num, channels[2]])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tmp1(x)
        x = F.relu(self.graph(x))
        x = self.tmp2(x)
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class OutputBlock(nn.Module):
    def __init__(self, Ko, c_in, channels, end_channel, node_num, dropout):
        super().__init__()
        self.tmp1 = TemporalConvLayer(Ko, c_in, channels[0])
        self.ln = nn.LayerNorm([node_num, channels[0]])
        self.fc1 = nn.Linear(channels[0], channels[1])
        self.fc2 = nn.Linear(channels[1], end_channel)

    def forward(self, x):
        x = self.tmp1(x)
        x = self.ln(x.permute(0, 2, 3, 1))
        x = F.relu(self.fc1(x))
        return self.fc2(x).permute(0, 3, 1, 2)


class STGCN(nn.Module):
    def __init__(self, adj, num_nodes, in_steps=12, out_steps=12, in_dim=1,
                 Kt=3, Ks=3, block_num=2, dropout=0.3):
        super().__init__()
        self.out_steps = out_steps
        adj = np.asarray(adj) - np.eye(num_nodes)
        adj[adj < 0] = 0
        gso = torch.tensor(normalize_adj(adj, "scalap")[0], dtype=torch.float32)

        blocks = [[in_dim]] + [[64, 16, 64] for _ in range(block_num)]
        Ko = in_steps - (Kt - 1) * 2 * block_num
        self.Ko = Ko
        if Ko > 0:
            blocks.append([128, 128])
        else:
            blocks.append([128])
        blocks.append([out_steps])

        self.st_blocks = nn.Sequential(*[
            STConvBlock(Kt, Ks, num_nodes, blocks[l][-1], blocks[l + 1], gso, dropout)
            for l in range(block_num)])
        if Ko > 0:
            self.output = OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0],
                                      num_nodes, dropout)
        else:
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0])
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0])

    def forward(self, x, label=None, batches_seen=None, task_level=None):
        x = x[..., :1].permute(0, 3, 1, 2)
        x = self.st_blocks(x)
        if self.Ko > 0:
            x = self.output(x)
        else:
            x = F.relu(self.fc1(x.permute(0, 2, 3, 1)))
            x = self.fc2(x).permute(0, 3, 1, 2)
        return x.transpose(2, 3)
