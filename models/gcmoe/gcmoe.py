import glob
import os

import torch
import torch.nn as nn

from common.metrics import masked_mae_loss
from .graph_features import compute_graph_features
from .router import GraphConditionedExpertRouter, GraphConditionedOutputRefinement, \
    compute_load_balance_loss

RUNS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "runs")


def _find_checkpoint(model_name, dataset):
    pattern = os.path.join(RUNS_ROOT, f"{model_name}_{dataset}", "*")
    dirs = sorted(glob.glob(pattern))
    for d in reversed(dirs):
        ckpt = os.path.join(d, "best.pt")
        if os.path.isfile(ckpt):
            return ckpt
    raise FileNotFoundError(
        f"No pretrained checkpoint found for {model_name} on {dataset} under {pattern}. "
        f"GC-MoE requires frozen pretrained GWNET/STGCN/AGCRN experts.")


def _build_gwnet(adj, num_nodes, device):
    from common.graph_utils import normalize_adj
    from .gwnet_expert import GWNET
    supports = [torch.tensor(a).to(device) for a in normalize_adj(adj, "doubletransition")]
    return GWNET(supports=supports, adp_adj=1, dropout=0.3, residual_channels=32,
                dilation_channels=32, skip_channels=256, end_channels=512,
                kernel_size=2, blocks=4, layers=2, node_num=num_nodes, input_dim=2,
                output_dim=1, seq_len=12, horizon=12)


def _build_stgcn(adj, num_nodes, device):
    from models.stgcn import STGCN
    return STGCN(adj=adj, num_nodes=num_nodes, in_steps=12, out_steps=12,
                in_dim=1, Kt=3, Ks=3, block_num=2, dropout=0.3)


def _build_agcrn(adj, num_nodes, device):
    from models.agcrn import AGCRN
    return AGCRN(node_num=num_nodes, input_dim=1, output_dim=1, seq_len=12, horizon=12,
                embed_dim=10, rnn_unit=64, num_layer=2, cheb_k=2)


_EXPERT_BUILDERS = {"GWNET": _build_gwnet, "STGCN": _build_stgcn, "AGCRN": _build_agcrn}


class GCMoE(nn.Module):
    loss_is_aux = True

    def __init__(self, adj, num_nodes, dataset, device, in_dim=3,
                 router_embed_dim=32, refine_hidden_dim=32, temperature=1.0,
                 noise_scale=0.5, load_balance_weight=0.001, routing_entropy_weight=0.5,
                 expert_names=("GWNET", "STGCN", "AGCRN")):
        super().__init__()
        self.load_balance_weight = load_balance_weight
        self.routing_entropy_weight = routing_entropy_weight
        self.expert_names = list(expert_names)
        self.scaler = None

        self.experts = nn.ModuleList()
        for name in self.expert_names:
            model = _EXPERT_BUILDERS[name](adj, num_nodes, device)
            ckpt = _find_checkpoint(name, dataset)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
            self.experts.append(model)

        graph_feats = compute_graph_features(adj, num_nodes).to(device)
        self.router = GraphConditionedExpertRouter(
            num_experts=len(self.experts), node_num=num_nodes, graph_features=graph_feats,
            adj_mx=adj, input_dim=in_dim, embed_dim=router_embed_dim,
            temperature=temperature, noise_scale=noise_scale, device=device)
        self.output_refine = GraphConditionedOutputRefinement(
            node_num=num_nodes, output_dim=1, adj_mx=adj, graph_features=graph_feats,
            routing_dim=len(self.experts), hidden_dim=refine_hidden_dim, device=device)

    def set_scaler(self, scaler):
        self.scaler = scaler

    def train(self, mode=True):
        super().train(mode)
        for e in self.experts:
            e.eval()
        return self

    def forward(self, x, label=None, batches_seen=None, task_level=None):
        routing_weights = self.router(x)

        expert_outputs = []
        with torch.no_grad():
            for expert in self.experts:
                expert_outputs.append(expert(x))
        stacked = torch.stack(expert_outputs, dim=0)
        stacked = stacked.permute(1, 3, 0, 2, 4)      
        rw = routing_weights.unsqueeze(-1).unsqueeze(-1)
        weighted = (stacked * rw).sum(dim=2)
        out = weighted.transpose(1, 2)

        out = self.output_refine(out, routing_weights=routing_weights)

        if self.training:
            pred = self.scaler.inverse_transform(out)
            true = self.scaler.inverse_transform(label[..., :1])
            mae_loss = masked_mae_loss(pred, true, null_val=0.0)
            if self.load_balance_weight > 0:
                balance_loss = compute_load_balance_loss(
                    routing_weights, entropy_weight=self.routing_entropy_weight)
                total_loss = mae_loss + self.load_balance_weight * balance_loss
            else:
                total_loss = mae_loss
            return out, total_loss
        return out
