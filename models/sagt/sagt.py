import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stemgnn.stemgnn import StockBlockLayer


class SpectralAdaptiveGT(nn.Module):
    def __init__(self, num_nodes, in_steps=12, out_steps=12, multi_layer=5,
                 dropout=0.5, embed_dim=10, k=32, step="A", device="cpu"):
        super().__init__()
        self.num_nodes = num_nodes
        self.time_step = in_steps
        self.horizon = out_steps
        self.stack_cnt = 2
        self.step = step
        self.k = k

        if step == "A":
            self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim))
        elif step in ("B", "C"):
            self.basis_W = nn.Parameter(torch.randn(num_nodes, k))
            if step == "B":
                self.theta = nn.Parameter(torch.randn(4, k) * 0.5)
            else:
                self.theta_mlp = nn.Sequential(
                    nn.Linear(in_steps, 32), nn.ReLU(), nn.Linear(32, 4 * k))
        else:
            raise ValueError(f"step {step} not in this module (A/B/C)")

        self.stock_block = nn.ModuleList([
            StockBlockLayer(self.time_step, num_nodes, multi_layer, stack_cnt=i)
            for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(self.time_step, self.time_step),
            nn.LeakyReLU(),
            nn.Linear(self.time_step, self.horizon),
        )
        self.dropout = nn.Dropout(dropout)
        self.to(device)

    def learned_adj(self):
        return F.softmax(F.relu(torch.mm(self.node_emb, self.node_emb.t())), dim=1)

    def get_laplacian(self, graph):
        D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
        return torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) \
            - torch.mm(torch.mm(D, graph), D)

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)
        lap = laplacian.unsqueeze(0)
        first = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second = lap
        third = 2 * torch.matmul(lap, second) - first
        forth = 2 * torch.matmul(lap, third) - second
        return torch.cat([first, second, third, forth], dim=0)

    def _basis(self):
        return torch.linalg.qr(self.basis_W, mode="reduced")[0]

    def compute_mul_L(self, x):
        if self.step == "A":
            return self.cheb_polynomial(self.get_laplacian(self.learned_adj()))
        U = self._basis()
        if self.step == "B":
            theta = self.theta
            return torch.einsum("nk,jk,mk->jnm", U, theta, U)
        ctx = x.mean(dim=2)
        theta = self.theta_mlp(ctx).view(-1, 4, self.k)
        return torch.einsum("nk,bjk,mk->bjnm", U, theta, U)

    def forward(self, x, label=None, batches_seen=None, task_level=None):
        x = x[..., 0]
        mul_L = self.compute_mul_L(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        for i in range(self.stack_cnt):
            forecast, X = self.stock_block[i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]
        forecast = self.fc(forecast)
        return forecast.permute(0, 2, 1).contiguous().unsqueeze(-1)
