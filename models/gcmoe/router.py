import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(adj_mx, node_num, device):
    adj = torch.as_tensor(adj_mx, dtype=torch.float32)
    if adj.shape[0] != node_num:
        adj = adj[:node_num, :node_num]
    adj = adj + torch.eye(node_num)
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (adj / row_sum).to(device)


class GraphConditionedExpertRouter(nn.Module):
    def __init__(self, num_experts, node_num, graph_features, adj_mx, input_dim,
                 embed_dim=32, temperature=1.0, noise_scale=0.5, device="cpu"):
        super().__init__()
        self.num_experts = num_experts
        self.node_num = node_num
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.top_k = num_experts  # D2 uses full soft routing, no top-k restriction

        self.register_buffer("graph_features", graph_features)
        num_graph_features = graph_features.shape[1]

        self.feature_projection = nn.Sequential(
            nn.Linear(num_graph_features, embed_dim), nn.LayerNorm(embed_dim),
            nn.ReLU(), nn.Dropout(0.1))
        self.node_embedding = nn.Parameter(torch.randn(node_num, embed_dim) * 0.01)
        self.static_routing = nn.Parameter(torch.randn(node_num, num_experts) * 0.01)

        self.register_buffer("static_norm_adj", build_norm_adj(adj_mx, node_num, device))
        self.graph_neighbor_gate = nn.Parameter(torch.tensor(0.5))

        dynamic_dim = 32
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, dynamic_dim), nn.ReLU(), nn.Dropout(0.1))
        self.temporal_attn = nn.Sequential(
            nn.Linear(dynamic_dim, dynamic_dim // 2), nn.ReLU(),
            nn.Linear(dynamic_dim // 2, 1))
        self.register_buffer("norm_adj", build_norm_adj(adj_mx, node_num, device))
        self.spatial_mixer = nn.Linear(dynamic_dim, dynamic_dim)
        self.dynamic_proj = nn.Sequential(
            nn.Linear(dynamic_dim, embed_dim), nn.ReLU(), nn.Dropout(0.1))
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1), nn.Sigmoid())
        self.dynamic_router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, num_experts))

        with torch.no_grad():
            val = 1.0 / num_experts
            for i in range(num_experts):
                pref = torch.full((node_num,), val)
                noise = torch.randn(node_num) * (val * 0.25)
                self.static_routing[:, i] = pref + noise

    def _noisy_topk(self, logits, training):
        if self.top_k >= self.num_experts:
            return F.softmax(logits / self.temperature, dim=-1)
        if training:
            logits = logits + torch.randn_like(logits) * self.noise_scale
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(-1, topk_idx, topk_vals)
        return F.softmax(masked / self.temperature, dim=-1)

    def forward(self, x):  # x: (B, T, N, D)
        B = x.shape[0]

        graph_feat = self.feature_projection(self.graph_features)
        static_repr = self.node_embedding + graph_feat
        neighbor_repr = torch.matmul(self.static_norm_adj, static_repr)
        static_repr = static_repr + torch.sigmoid(self.graph_neighbor_gate) * neighbor_repr
        static_logits = self.static_routing

        h = self.input_encoder(x)
        attn_weights = torch.softmax(self.temporal_attn(h), dim=1)
        node_input_repr = (h * attn_weights).sum(dim=1)
        neighbor_repr = torch.matmul(self.norm_adj, node_input_repr)
        dynamic_repr = node_input_repr + self.spatial_mixer(neighbor_repr)
        dynamic_repr = self.dynamic_proj(dynamic_repr)

        static_expanded = static_repr.unsqueeze(0).expand(B, -1, -1)
        gate_weight = self.gate(torch.cat([static_expanded, dynamic_repr], dim=-1))
        fused_repr = gate_weight * static_expanded + (1 - gate_weight) * dynamic_repr

        dynamic_logits = self.dynamic_router(fused_repr)
        final_logits = static_logits.unsqueeze(0) + dynamic_logits
        return self._noisy_topk(final_logits, self.training)  # (B, N, num_experts)


class GraphConditionedOutputRefinement(nn.Module):
    def __init__(self, node_num, output_dim, adj_mx, graph_features, routing_dim,
                 hidden_dim=32, dropout=0.1, device="cpu"):
        super().__init__()
        self.node_num = node_num
        self.output_dim = output_dim
        self.routing_dim = routing_dim

        self.register_buffer("graph_feats", graph_features)
        input_dim = graph_features.shape[1] + routing_dim
        self.register_buffer("norm_adj", build_norm_adj(adj_mx, node_num, device))

        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim))
        self.bias_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim))
        self.spatial_refine = nn.Linear(output_dim, output_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(-5.0))

        nn.init.zeros_(self.scale_net[-1].weight); nn.init.zeros_(self.scale_net[-1].bias)
        nn.init.zeros_(self.bias_net[-1].weight); nn.init.zeros_(self.bias_net[-1].bias)
        if output_dim == self.spatial_refine.weight.shape[-1]:
            nn.init.eye_(self.spatial_refine.weight)
        else:
            nn.init.xavier_uniform_(self.spatial_refine.weight, gain=0.01)

    def forward(self, x, routing_weights):  # x: (B, T, N, D)
        B, T, N, D = x.shape
        rw_avg = routing_weights.detach().mean(dim=0)  # (N, num_experts)
        cond = torch.cat([self.graph_feats, rw_avg], dim=-1)

        scale = 0.3 * torch.tanh(self.scale_net(cond))
        bias = 0.3 * torch.tanh(self.bias_net(cond))
        correction = x * scale.unsqueeze(0).unsqueeze(0) + bias.unsqueeze(0).unsqueeze(0)

        corr_flat = correction.reshape(B * T, N, D)
        spatial = self.spatial_refine(torch.matmul(self.norm_adj, corr_flat))
        spatial = spatial.reshape(B, T, N, D)
        return x + torch.sigmoid(self.gate) * spatial


def compute_load_balance_loss(routing_weights, entropy_weight=0.5):
    E = routing_weights.shape[-1]
    expert_indices = routing_weights.argmax(dim=-1)
    one_hot = F.one_hot(expert_indices, num_classes=E).float()
    f_e = one_hot.mean(dim=[0, 1])
    P_e = routing_weights.mean(dim=[0, 1])
    balance_loss = E * (f_e * P_e).sum()

    importance_std = P_e.std()
    importance_mean = P_e.mean()
    cv2 = (importance_std / (importance_mean + 1e-8)) ** 2
    balance_loss = balance_loss + 0.1 * cv2

    entropy = -(routing_weights * torch.log(routing_weights + 1e-9)).sum(dim=-1)
    balance_loss = balance_loss + entropy_weight * entropy.mean()
    return balance_loss
