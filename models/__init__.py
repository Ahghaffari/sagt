

def build_model(cfg, adj, num_nodes, device):
    name = cfg["model_name"]
    m = cfg["model"]
    d = cfg["data"]
    seq_len, horizon = d["in_steps"], d["out_steps"]
    in_dim = m.get("in_dim", 1)
    common = dict(node_num=num_nodes, input_dim=in_dim, output_dim=1,
                  seq_len=seq_len, horizon=horizon)

    if name == "STGCN":
        from .stgcn import STGCN
        return STGCN(adj=adj, num_nodes=num_nodes, in_steps=seq_len, out_steps=horizon,
                     in_dim=in_dim, Kt=m["Kt"], Ks=m["Ks"],
                     block_num=m["block_num"], dropout=m["dropout"])

    if name == "AGCRN":
        from .agcrn import AGCRN
        return AGCRN(embed_dim=m["embed_dim"], rnn_unit=m["rnn_unit"],
                     num_layer=m["num_layer"], cheb_k=m["cheb_k"], **common)

    if name == "DCRNN":
        from .dcrnn import DCRNN
        return DCRNN(device=device, adj_mx=adj, n_filters=m["n_filters"],
                     max_diffusion_step=m["max_diffusion_step"],
                     filter_type=m.get("filter_type", "doubletransition"),
                     num_rnn_layers=m["num_rnn_layers"],
                     cl_decay_steps=m["cl_decay_steps"], **common)

    if name == "StemGNN":
        from .stemgnn import StemGNN
        return StemGNN(units=num_nodes, stack_cnt=2, time_step=seq_len, horizon=horizon,
                       multi_layer=m["multi_layer"], dropout_rate=m["dropout"],
                       leaky_rate=m.get("leaky_rate", 0.2), device=device)

    if name == "STAEformer":
        from .staeformer import STAEformer
        return STAEformer(num_nodes=num_nodes, in_steps=seq_len, out_steps=horizon,
                          steps_per_day=m.get("steps_per_day", 288),
                          input_dim=in_dim, input_embedding_dim=m["input_embedding_dim"],
                          tod_embedding_dim=m["tod_embedding_dim"],
                          dow_embedding_dim=m["dow_embedding_dim"],
                          spatial_embedding_dim=m["spatial_embedding_dim"],
                          adaptive_embedding_dim=m["adaptive_embedding_dim"],
                          feed_forward_dim=m["feed_forward_dim"], num_heads=m["num_heads"],
                          num_layers=m["num_layers"], dropout=m["dropout"])

    if name == "TESTAM":
        from .testam import TESTAM
        return TESTAM(num_nodes=num_nodes, in_dim=in_dim, hidden_size=m["hidden_size"],
                      layers=m["layers"], dropout=m["dropout"],
                      prob_mul=m.get("prob_mul", False), seq_len=seq_len, horizon=horizon,
                      warmup_epochs=m.get("warmup_epochs", 0))

    if name == "PDFormer":
        from .pdformer import PDFormer
        return PDFormer(cfg, adj, num_nodes, device)

    if name == "MegaCRN":
        from .megacrn import MegaCRN
        return MegaCRN(num_nodes=num_nodes, input_dim=1, output_dim=1, horizon=horizon,
                       rnn_units=m.get("rnn_units", 64), num_layers=m.get("num_layers", 1),
                       cheb_k=m.get("cheb_k", 3), mem_num=m.get("mem_num", 20),
                       mem_dim=m.get("mem_dim", 64), cl_decay_steps=m.get("cl_decay_steps", 2000),
                       lamb=m.get("lamb", 0.01), lamb1=m.get("lamb1", 0.01))

    if name == "GCMoE":
        from .gcmoe import GCMoE
        return GCMoE(adj=adj, num_nodes=num_nodes, dataset=cfg["dataset"], device=device,
                    in_dim=in_dim, router_embed_dim=m.get("router_embed_dim", 32),
                    refine_hidden_dim=m.get("refine_hidden_dim", 32),
                    temperature=m.get("temperature", 1.0), noise_scale=m.get("noise_scale", 0.5),
                    load_balance_weight=m.get("load_balance_weight", 0.001),
                    routing_entropy_weight=m.get("routing_entropy_weight", 0.5))

    if name == "SAGT":
        step = m.get("step", "A")
        if step in ("A", "B", "C"):
            from .sagt import SpectralAdaptiveGT
            return SpectralAdaptiveGT(num_nodes=num_nodes, in_steps=seq_len, out_steps=horizon,
                                      multi_layer=m.get("multi_layer", 5), dropout=m.get("dropout", 0.5),
                                      embed_dim=m.get("embed_dim", 10), k=m.get("k", 32),
                                      step=step, device=device)
        from .sagt.transformer import SAGTransformer
        return SAGTransformer(
            num_nodes=num_nodes, in_steps=seq_len, out_steps=horizon, input_dim=in_dim,
            steps_per_day=m.get("steps_per_day", 288),
            input_embedding_dim=m.get("input_embedding_dim", 24),
            tod_embedding_dim=m.get("tod_embedding_dim", 24),
            dow_embedding_dim=m.get("dow_embedding_dim", 24),
            adaptive_embedding_dim=m.get("adaptive_embedding_dim", 80),
            feed_forward_dim=m.get("feed_forward_dim", 256), num_heads=m.get("num_heads", 4),
            num_layers=m.get("num_layers", 3), k=m.get("k", 32), dropout=m.get("dropout", 0.1),
            freq_bin_attn=(step in ("E", "G")), adj=adj, lap_init=m.get("lap_init", False),
            static_filter=m.get("static_filter", False), hybrid=(step in ("F", "G")),
            local_graph=m.get("local_graph", True),
            global_branch=m.get("global_branch", True), device=device)

    raise ValueError(f"unknown model {name}")
