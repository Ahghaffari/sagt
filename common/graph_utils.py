import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return sp.eye(adj.shape[0]) - d_mat.dot(adj).dot(d_mat).tocoo()


def calculate_scaled_laplacian(adj, lambda_max=2.0, undirected=True):
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format="csr", dtype=L.dtype)
    return ((2 / lambda_max * L) - I).astype(np.float32).todense()


def calculate_sym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat.dot(adj).dot(d_mat).astype(np.float32).todense()


def calculate_asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def normalize_adj(adj, adj_type, return_type="dense"):
    if adj_type == "scalap":
        mats = [calculate_scaled_laplacian(adj)]
    elif adj_type == "normlap":
        mats = [calculate_normalized_laplacian(adj)]
    elif adj_type == "symadj":
        mats = [calculate_sym_adj(adj)]
    elif adj_type == "transition":
        mats = [calculate_asym_adj(adj)]
    elif adj_type == "doubletransition":
        mats = [calculate_asym_adj(adj), calculate_asym_adj(adj.T)]
    elif adj_type == "identity":
        mats = [np.eye(adj.shape[0], dtype=np.float32)]
    else:
        raise ValueError(f"unknown adj_type {adj_type}")
    if return_type == "coo":
        return [sp.coo_matrix(np.asarray(m)).astype(np.float32) for m in mats]
    return [np.asarray(m).astype(np.float32) for m in mats]
