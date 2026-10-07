import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path

from .pdformer_core import PDFormerCore

PDF_CACHE = "/home/ubuntu/PDFormer/libcity/cache/dataset_cache"
OUR_CACHE = "/mnt/data/spectral_agt/cache/pdformer"
CACHE_MAP = {
    "PEMS04": ("dtw_PeMS04.npy", "pattern_keys_kshape_PeMS04_14_3_16_5.npy"),
    "PEMS07": ("dtw_PeMS07.npy", "pattern_keys_kshape_PeMS07_14_3_16_5.npy"),
    "PEMS08": ("dtw_PeMS08.npy", "pattern_keys_kshape_PeMS08_21_3_16_5.npy"),
    "METRLA": (f"{OUR_CACHE}/dtw_METRLA.npy", f"{OUR_CACHE}/pattern_keys_METRLA.npy"),
    "PEMSBAY": (f"{OUR_CACHE}/dtw_PEMSBAY.npy", f"{OUR_CACHE}/pattern_keys_PEMSBAY.npy"),
    "EXPYTKY": (f"{OUR_CACHE}/dtw_EXPYTKY.npy", f"{OUR_CACHE}/pattern_keys_EXPYTKY.npy"),
}


def _shortest_hops(adj):
    b = (adj > 0).astype(np.float64)
    sh = shortest_path(csgraph=b, method="D", directed=False)
    sh[np.isinf(sh)] = 511
    np.fill_diagonal(sh, 0)
    return sh.astype(np.float32)


def _laplacian_pe(adj, lape_dim):
    a = adj.copy().astype(np.float64)
    np.fill_diagonal(a, 0)
    d = a.sum(1)
    isolated = int((d == 0).sum())
    dinv = np.zeros_like(d)
    dinv[d > 0] = np.power(d[d > 0], -0.5)
    L = np.eye(a.shape[0]) - (dinv[:, None] * a * dinv[None, :])
    val, vec = np.linalg.eig(L)
    idx = val.argsort()
    vec = np.real(vec[:, idx])
    pe = vec[:, isolated + 1: lape_dim + isolated + 1]
    return pe.astype(np.float32)


class PDFormer(nn.Module):
    """Adapter: x (B,T,N,3)[flow,tod,dow] -> PDFormerCore (feature_dim=9:
    [flow, tod, dow_onehot(7)]); returns (B,T',N,1) normalized."""

    def __init__(self, cfg, adj, num_nodes, device):
        super().__init__()
        dataset = cfg["dataset"]
        if dataset not in CACHE_MAP:
            raise ValueError(f"PDFormer cache not available for {dataset}; "
                             f"precompute DTW + pattern keys first")
        m = cfg["model"]
        lape_dim = m.get("lape_dim", 8)
        dtw_file, pk_file = CACHE_MAP[dataset]
        dtw_matrix = np.load(dtw_file if os.path.isabs(dtw_file)
                             else os.path.join(PDF_CACHE, dtw_file))
        pattern_keys = np.load(pk_file if os.path.isabs(pk_file)
                               else os.path.join(PDF_CACHE, pk_file))

        sh_mx = _shortest_hops(adj)
        lap_mx = _laplacian_pe(adj, lape_dim)

        config = {
            "dataset": dataset, "device": device,
            "embed_dim": m.get("embed_dim", 64), "skip_dim": m.get("skip_dim", 256),
            "lape_dim": lape_dim, "geo_num_heads": m.get("geo_num_heads", 4),
            "sem_num_heads": m.get("sem_num_heads", 2), "t_num_heads": m.get("t_num_heads", 2),
            "mlp_ratio": m.get("mlp_ratio", 4), "qkv_bias": True,
            "drop": m.get("drop", 0.0), "attn_drop": m.get("attn_drop", 0.0),
            "drop_path": m.get("drop_path", 0.3), "s_attn_size": m.get("s_attn_size", 3),
            "t_attn_size": m.get("t_attn_size", 3), "enc_depth": m.get("enc_depth", 6),
            "type_ln": m.get("type_ln", "pre"), "type_short_path": "hop",
            "output_dim": 1, "input_window": cfg["data"]["in_steps"],
            "output_window": cfg["data"]["out_steps"],
            "add_time_in_day": True, "add_day_in_week": True,
            "far_mask_delta": m.get("far_mask_delta", 5), "dtw_delta": m.get("dtw_delta", 5),
            "use_curriculum_learning": False, "huber_delta": 1, "world_size": 1,
            "step_size": 0, "max_epoch": 1, "task_level": cfg["data"]["out_steps"],
        }
        data_feature = {
            "scaler": None, "num_nodes": num_nodes, "feature_dim": 9, "ext_dim": 8,
            "num_batches": 1, "dtw_matrix": dtw_matrix, "adj_mx": adj,
            "sd_mx": None, "sh_mx": sh_mx, "pattern_keys": pattern_keys,
        }
        self.core = PDFormerCore(config, data_feature)
        self.register_buffer("lap_mx", torch.from_numpy(lap_mx))

    def _to_pdformer_input(self, x):
        flow = x[..., :1]
        tod = x[..., 1:2]
        dow = x[..., 2].long().clamp(0, 6)
        dow_oh = F.one_hot(dow, num_classes=7).float()
        return torch.cat([flow, tod, dow_oh], dim=-1)

    def forward(self, x, label=None, batches_seen=None, task_level=None):
        x9 = self._to_pdformer_input(x)
        out = self.core({"X": x9}, self.lap_mx)
        return out
