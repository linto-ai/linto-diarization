import numpy as np
import scipy
from sklearn.cluster import SpectralClustering
import torch

# NME low-level operations
# These functions are taken from the Kaldi scripts.

# Prepares binarized(0/1) affinity matrix with p_neighbors non-zero elements in each row
def get_kneighbors_conn(X_dist, p_neighbors):
    X_dist_out = np.zeros_like(X_dist)
    for i, line in enumerate(X_dist):
        sorted_idx = np.argsort(line)
        sorted_idx = sorted_idx[::-1]
        indices = sorted_idx[:p_neighbors]
        X_dist_out[indices, i] = 1
    return X_dist_out


# Thresolds affinity matrix to leave p maximum non-zero elements in each row
def Threshold(A, p):
    N = A.shape[0]
    Ap = np.zeros((N, N))
    for i in range(N):
        thr = sorted(A[i, :], reverse=True)[p]
        Ap[i, A[i, :] > thr] = A[i, A[i, :] > thr]
    return Ap


# Computes Laplacian of a matrix
def Laplacian(A):
    d = np.sum(A, axis=1) - np.diag(A)
    D = np.diag(d)
    return D - A


# Calculates eigengaps (differences between adjacent eigenvalues sorted in descending order)
def Eigengap(S):
    S = sorted(S)
    return np.diff(S)

def getLamdaGaplist(lambdas):
    lambdas = np.real(lambdas)
    return list(lambdas[1:] - lambdas[:-1])

# Computes parameters of normalized eigenmaps for automatic thresholding selection
def ComputeNMEParameters(A, p, max_num_clusters, device):
    # p-Neighbour binarization
    Ap = get_kneighbors_conn(A, p)
    # Symmetrization
    Ap = (Ap + np.transpose(Ap)) / 2
    # Laplacian matrix computation
    Lp = Laplacian(Ap)
    # Get max_num_clusters+1 smallest eigenvalues
    from torch.linalg import eigh     
    
    if device=="cuda" and torch.cuda.is_available()== True:       
        Lp = torch.from_numpy(Lp).float().to('cuda')
        lambdas, _ = eigh(Lp)
        S = lambdas.cpu().numpy()
        
    else:
        Lp = torch.from_numpy(Lp).float()
        lambdas, _ = eigh(Lp)
        S = lambdas.cpu().numpy()
    
    # Eigengap computation
    
    e = np.sort(S)
    g = getLamdaGaplist(e)
    k = np.argmax(g[: min(max_num_clusters, len(g))]) 
    arg_sorted_idx = np.argsort(g[: max_num_clusters])[::-1]
    max_key = arg_sorted_idx[0]
    max_eig_gap = g[max_key] / (max(e) + 1e-10)
    r = (p / A.shape[0]) / (max_eig_gap + 1e-10)
    
    
    return (e, g, k, r)


"""
Performs spectral clustering with Normalized Maximum Eigengap (NME)
Parameters:
   A: affinity matrix (matrix of pairwise cosine similarities or PLDA scores between speaker embeddings)
   num_clusters: number of clusters to generate (if None, determined automatically)
   max_num_clusters: maximum allowed number of clusters to generate
   pmax: maximum count for matrix binarization (should be at least 2)
   pbest: best count for matrix binarization (if 0, determined automatically)
Returns: cluster assignments for every speaker embedding   
"""


def NME_SpectralClustering(
    A, num_clusters=None, max_num_clusters=None, pbest=0, pmin=3, pmax=20, device=None
):
    if max_num_clusters is None:
        assert num_clusters is not None, "Cannot have both num_clusters and max_num_clusters be None"
        max_num_clusters = num_clusters

    if pbest == 0:
        print("Selecting best number of neighbors for affinity matrix thresolding:")
        rbest = None
        kbest = None
        for p in range(pmin, pmax + 1):
            e, g, k, r = ComputeNMEParameters(A, p, max_num_clusters,device)
            if rbest is None or rbest > r:
                rbest = r
                pbest = p
                kbest = k        
        num_clusters = num_clusters if num_clusters is not None else (kbest + 1)
        return NME_SpectralClustering_sklearn(
            A, num_clusters, pbest
        )

    if num_clusters is None:
        e, g, k, r = ComputeNMEParameters(A, pbest, max_num_clusters)
        return NME_SpectralClustering_sklearn(A, k + 1, pbest)

    return NME_SpectralClustering_sklearn(A, num_clusters, pbest)


"""
Performs spectral clustering with Normalized Maximum Eigengap (NME) with fixed threshold and number of clusters
Parameters:
   A: affinity matrix (matrix of pairwise cosine similarities or PLDA scores between speaker embeddings)
   OLVec: 0/1 vector denoting which segments are overlap segments
   num_clusters: number of clusters to generate
   pbest: best count for matrix binarization
Returns: cluster assignments for every speaker embedding   
"""


def NME_SpectralClustering_sklearn(A, num_clusters, pbest):
    
    # Ap = Threshold(A, pbest)
    Ap = get_kneighbors_conn(A, pbest)  # thresholded and binarized
    Ap = (Ap + np.transpose(Ap)) / 2
    
    
    model = SpectralClustering(
        n_clusters=num_clusters, affinity="precomputed", random_state=0
    )
    labels = model.fit_predict(Ap)
    return labels
