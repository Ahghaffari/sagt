import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.staeformer.staeformer import SelfAttentionLayer


def laplacian_eigvecs(adj, k):
    a = np.maximum(adj, adj.T).astype(np.float64)
    np.fill_diagonal(a, 0)
    d = a.sum(1)
    dinv = np.zeros_like(d)
    dinv[d > 0] = d[d > 0] ** -0.5
    L = np.eye(len(a)) - dinv[:, None] * a * dinv[None, :]
    _, vecs = np.linalg.eigh(L)
    return torch.tensor(vecs[:, :k], dtype=torch.float32)


class SpectralSpatialLayer(nn.Module):
    def __init__(self, num_nodes, k, model_dim, in_steps, feed_forward_dim=256,
                 dropout=0.1, basis_init=None, static_filter=False):
        super().__init__()
        self.k = k
        if basis_init is not None:
            self.basis_W = nn.Parameter(basis_init.clone() + 0.01 * torch.randn(num_nodes, k))
        else:
            self.basis_W = nn.Parameter(torch.randn(num_nodes, k))
        self.theta_static = nn.Parameter(torch.ones(k, model_dim)) if static_filter else None
        self.filter_mlp = nn.Sequential(
            nn.Linear(in_steps * model_dim, 64), nn.ReLU(), nn.Linear(64, k))
        self.proj = nn.Linear(model_dim, model_dim)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim), nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim))
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, d = x.shape
        U = torch.linalg.qr(self.basis_W, mode="reduced")[0]
        x_hat = torch.einsum("nk,btnd->btkd", U, x)
        ctx = x.mean(dim=2).reshape(B, T * d)
        filt = torch.sigmoid(self.filter_mlp(ctx)).view(B, 1, self.k, 1)
        x_hat = x_hat * filt
        if self.theta_static is not None:
            x_hat = x_hat * self.theta_static
        out = torch.einsum("nk,btkd->btnd", U, x_hat)
        out = self.proj(out)
        x = self.ln1(x + self.dropout1(out))
        return self.ln2(x + self.dropout2(self.ffn(x)))


class HybridSpectralSpatialLayer(nn.Module):
    def __init__(self, num_nodes, k, model_dim, feed_forward_dim=256, dropout=0.1,
                 supports=None, local_graph=True, global_branch=True):
        super().__init__()
        self.k = k
        self.local_graph = local_graph and supports is not None and len(supports) > 0
        self.global_branch = global_branch
        if self.global_branch:
            self.basis_Wq = nn.Parameter(torch.randn(num_nodes, k))
            self.basis_Wk = nn.Parameter(torch.randn(num_nodes, k))
            self.filter_mlp = nn.Sequential(
                nn.Linear(model_dim, 64), nn.ReLU(), nn.Linear(64, k))
        if self.local_graph:
            for i, a in enumerate(supports):
                self.register_buffer(f"support_{i}", a, persistent=False)
            self.n_supports = len(supports)
            self.local_lins = nn.ModuleList(
                [nn.Linear(model_dim, model_dim, bias=False) for _ in supports])
        self.fuse = nn.Linear(2 * model_dim, model_dim)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim), nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim))
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, d = x.shape
        g = x.new_zeros(B, T, N, d)
        if self.global_branch:
            Uq = torch.linalg.qr(self.basis_Wq, mode="reduced")[0]
            Uk = torch.linalg.qr(self.basis_Wk, mode="reduced")[0]
            theta = self.filter_mlp(x.mean(dim=2))
            x_hat = torch.einsum("nk,btnd->btkd", Uk, x) * theta.unsqueeze(-1)
            g = torch.einsum("nk,btkd->btnd", Uq, x_hat)
        loc = x.new_zeros(B, T, N, d)
        if self.local_graph:
            xf = x.permute(2, 0, 1, 3).reshape(N, B * T * d)
            for i in range(self.n_supports):
                A = getattr(self, f"support_{i}")
                y = torch.sparse.mm(A, xf).reshape(N, B, T, d).permute(1, 2, 0, 3)
                loc = loc + self.local_lins[i](y)
        out = self.fuse(torch.cat([loc, g], dim=-1))
        x = self.ln1(x + self.dropout1(out))
        return self.ln2(x + self.dropout2(self.ffn(x)))


class FreqBinAttention(nn.Module):
    def __init__(self, num_nodes, in_steps, node_dim=16):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.n_bins = in_steps // 2 + 1
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + in_steps, 32), nn.ReLU(), nn.Linear(32, self.n_bins))
        self.last_attn = None

    def forward(self, x):
        B, T, N = x.shape
        ctx = x.transpose(1, 2)
        node_emb_b = self.node_emb.unsqueeze(0).expand(B, -1, -1)
        feat = torch.cat([node_emb_b, ctx], dim=-1)
        attn = torch.softmax(self.mlp(feat), dim=-1)
        self.last_attn = attn.detach()
        w = self.n_bins * attn
        xf = torch.fft.rfft(x, dim=1)
        xf = xf * w.transpose(1, 2)
        return torch.fft.irfft(xf, n=T, dim=1)


class SAGTransformer(nn.Module):
    def __init__(self, num_nodes, in_steps=12, out_steps=12, input_dim=3,
                 steps_per_day=288, input_embedding_dim=24, tod_embedding_dim=24,
                 dow_embedding_dim=24, adaptive_embedding_dim=80, feed_forward_dim=256,
                 num_heads=4, num_layers=3, k=32, dropout=0.1, freq_bin_attn=False,
                 adj=None, lap_init=False, static_filter=False, hybrid=False,
                 local_graph=True, global_branch=True, device="cpu"):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.steps_per_day = steps_per_day
        self.tod_dim = tod_embedding_dim
        self.dow_dim = dow_embedding_dim
        self.freq_bin_attn = freq_bin_attn

        self.model_dim = (input_embedding_dim + tod_embedding_dim
                          + dow_embedding_dim + adaptive_embedding_dim)
        if freq_bin_attn:
            self.freq_attn = FreqBinAttention(num_nodes, in_steps)
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim)))

        basis_init = None
        if lap_init and adj is not None:
            basis_init = laplacian_eigvecs(np.asarray(adj), k)

        self.temporal_layers = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)])
        if hybrid:
            supports = None
            if local_graph and adj is not None:
                from common.graph_utils import normalize_adj
                mats = normalize_adj(np.asarray(adj), "doubletransition")
                supports = [torch.tensor(m).to_sparse_coo() for m in mats]
            self.spatial_layers = nn.ModuleList([
                HybridSpectralSpatialLayer(num_nodes, k, self.model_dim,
                                           feed_forward_dim=feed_forward_dim, dropout=dropout,
                                           supports=supports, local_graph=local_graph,
                                           global_branch=global_branch)
                for _ in range(num_layers)])
        else:
            self.spatial_layers = nn.ModuleList([
                SpectralSpatialLayer(num_nodes, k, self.model_dim, in_steps,
                                     feed_forward_dim=feed_forward_dim, dropout=dropout,
                                     basis_init=basis_init, static_filter=static_filter)
                for _ in range(num_layers)])

        self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps)
        self.to(device)

    def forward(self, x, label=None, batches_seen=None, task_level=None):
        B = x.shape[0]
        tod = x[..., 1]
        dow = x[..., 2]
        flow = x[..., :self.input_dim]
        if self.freq_bin_attn:
            flow = flow.clone()
            flow[..., 0] = self.freq_attn(x[..., 0])

        feats = [self.input_proj(flow)]
        if self.tod_dim > 0:
            feats.append(self.tod_embedding((tod * self.steps_per_day).long()))
        if self.dow_dim > 0:
            feats.append(self.dow_embedding(dow.long()))
        feats.append(self.adaptive_embedding.expand(B, *self.adaptive_embedding.shape))
        z = torch.cat(feats, dim=-1)

        for t_layer, s_layer in zip(self.temporal_layers, self.spatial_layers):
            z = t_layer(z, dim=1)
            z = s_layer(z)
        out = z.transpose(1, 2).reshape(B, self.num_nodes, self.in_steps * self.model_dim)
        out = self.output_proj(out).transpose(1, 2)
        return out.unsqueeze(-1)
