import numpy as np
import torch

_CACHE = {}


def compute_graph_features(adj, node_num):
    import networkx as nx
    from scipy.sparse.linalg import eigsh as sp_eigsh

    key = (id(adj), node_num)
    if key in _CACHE:
        return _CACHE[key]

    adj_np = np.asarray(adj)
    if adj_np.shape[0] != node_num:
        adj_np = adj_np[:node_num, :node_num]

    G = nx.from_numpy_array(adj_np)
    feats = []

    def _norm(arr):
        arr = np.array(arr, dtype=np.float32)
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / (rng + 1e-8)

    degree = np.array([G.degree(i, weight="weight") for i in range(node_num)], dtype=np.float32)
    feats.append(_norm(degree))


    close = np.array([nx.closeness_centrality(G)[i] for i in range(node_num)], dtype=np.float32)
    feats.append(_norm(close))
    
    clust = np.array([nx.clustering(G, weight="weight")[i] for i in range(node_num)], dtype=np.float32)

    feats.append(_norm(clust))


    pr = nx.pagerank(G, weight="weight")
    pr_arr = np.array([pr[i] for i in range(node_num)], dtype=np.float32)
    feats.append(_norm(pr_arr))


    k = None if node_num <= 500 else min(500, node_num)
    seed = None if k is None else 42
    btwn = nx.betweenness_centrality(G, k=k, seed=seed, weight="weight", normalized=True)
    btwn_arr = np.array([btwn[i] for i in range(node_num)], dtype=np.float32)

    feats.append(_norm(btwn_arr))

    G_simple = nx.Graph(G)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    core = nx.core_number(G_simple)
    core_arr = np.array([core[i] for i in range(node_num)], dtype=np.float32)

    feats.append(_norm(core_arr))


    ec = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    ec_arr = np.array([ec[i] for i in range(node_num)], dtype=np.float32)

    feats.append(_norm(ec_arr))


    L = nx.normalized_laplacian_matrix(G, weight="weight").astype(np.float64)
    vals, vecs = np.linalg.eigh(L.toarray())
    vecs = vecs[:, np.argsort(vals)]
    skip = nx.number_connected_components(G)
    z = np.zeros(node_num, dtype=np.float32)
    fiedler = vecs[:, skip].astype(np.float32)     if vecs.shape[1] > skip     else z
    spec3   = vecs[:, skip + 1].astype(np.float32) if vecs.shape[1] > skip + 1 else z
    fiedler = -fiedler if fiedler[np.argmax(np.abs(fiedler))] < 0 else fiedler
    spec3   = -spec3   if spec3[np.argmax(np.abs(spec3))]     < 0 else spec3

    feats.append(_norm(fiedler))
    feats.append(_norm(spec3))

    result = torch.FloatTensor(np.stack(feats, axis=1))
    _CACHE[key] = result
    return result
