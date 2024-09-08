# AUTHORS
# Jose PATINO, EURECOM, Sophia-Antipolis, France, 2019
# http://www.eurecom.fr/en/people/patino-jose
# Contact: patino[at]eurecom[dot]fr, josempatinovillar[at]gmail[dot]com

import numpy as np
import scipy
import scipy.sparse as sparse
import sklearn
from scipy import sparse
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from sklearn import mixture
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array, check_random_state, check_symmetric
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.validation import check_array

__all__ = [
    "py_webrtcvad",
    "getSegmentTable",
    "trainKBM",
    "getVgMatrix",
    "getSegmentBKs",
    "performClusteringLinkage",
    "getSpectralClustering",
    "performResegmentation",
]


def py_webrtcvad(data, fs, fs_vad, hoplength=30, vad_mode=0):
    import webrtcvad
    from librosa.core import resample
    from librosa.util import frame

    """ Voice activity detection.
    This was implementioned for easier use of py-webrtcvad.
    Thanks to: https://github.com/wiseman/py-webrtcvad.git
    Parameters
    ----------
    data : ndarray
        numpy array of mono (1 ch) speech data.
        1-d or 2-d, if 2-d, shape must be (1, time_length) or (time_length, 1).
        if data type is int, -32768 < data < 32767.
        if data type is float, -1 < data < 1.
    fs : int
        Sampling frequency of data.
    fs_vad : int, optional
        Sampling frequency for webrtcvad.
        fs_vad must be 8000, 16000, 32000 or 48000.
        Default is 16000.
    hoplength : int, optional
        Step size[milli second].
        hoplength must be 10, 20, or 30.
        Default is 0.1.
    vad_mode : int, optional
        set vad aggressiveness.
        As vad_mode increases, it becomes more aggressive.
        vad_mode must be 0, 1, 2 or 3.
        Default is 0.
    Returns
    -------
    vact : ndarray
        voice activity. time length of vact is same as input data.
        If 0, it is unvoiced, 1 is voiced.
    """

    # check argument
    if fs_vad not in [8000, 16000, 32000, 48000]:
        raise ValueError("fs_vad must be 8000, 16000, 32000 or 48000.")

    if hoplength not in [10, 20, 30]:
        raise ValueError("hoplength must be 10, 20, or 30.")

    if vad_mode not in [0, 1, 2, 3]:
        raise ValueError("vad_mode must be 0, 1, 2 or 3.")

    # check data
    if data.dtype.kind == "i":
        if data.max() > 2**15 - 1 or data.min() < -(2**15):
            raise ValueError(
                "when data type is int, data must be -32768 < data < 32767."
            )
        data = data.astype("f")

    elif data.dtype.kind == "f":
        if np.abs(data).max() >= 1:
            data = data / np.abs(data).max() * 0.9
            print("Warning: input data was rescaled.")
        data = (data * 2**15).astype("f")
    else:
        raise ValueError("data dtype must be int or float.")

    data = data.squeeze()
    if not data.ndim == 1:
        raise ValueError("data must be mono (1 ch).")

    # resampling
    if fs != fs_vad:
        resampled = resample(data, fs, fs_vad)
    else:
        resampled = data

    resampled = resampled.astype("int16")

    hop = fs_vad * hoplength // 1000
    framelen = resampled.size // hop + 1
    padlen = framelen * hop - resampled.size
    paded = np.lib.pad(resampled, (0, padlen), "constant", constant_values=0)
    framed = frame(paded, frame_length=hop, hop_length=hop).T

    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    valist = [vad.is_speech(tmp.tobytes(), fs_vad) for tmp in framed]

    hop_origin = fs * hoplength // 1000
    va_framed = np.zeros([len(valist), hop_origin])
    va_framed[valist] = 1

    return va_framed.reshape(-1)[: data.size]


def get_py_webrtcvad_segments(vad_info, fs):
    vad_index = np.where(vad_info == 1.0)  # find the speech index
    vad_diff = np.diff(vad_index)

    vad_temp = np.zeros_like(vad_diff)
    vad_temp[np.where(vad_diff == 1)] = 1
    vad_temp = np.column_stack((np.array([0]), vad_temp, np.array([0])))
    final_index = np.diff(vad_temp)

    starts = np.where(final_index == 1)
    ends = np.where(final_index == -1)

    sad_info = np.column_stack((starts[1], ends[1]))
    vad_index = vad_index[0]

    segments = np.zeros_like(sad_info, dtype=np.float32)
    for i in range(sad_info.shape[0]):
        segments[i][0] = float(vad_index[sad_info[i][0]]) / fs
        segments[i][1] = float(vad_index[sad_info[i][1]] + 1) / fs

    return segments  # present in seconds


def getSegmentTable(mask, speechMapping, wLength, wIncr, wShift):
    changePoints, segBeg, segEnd, nSegs = unravelMask(mask)
    segmentTable = np.empty([0, 4])
    for i in range(nSegs):
        begs = np.arange(segBeg[i], segEnd[i], wShift)
        bbegs = np.maximum(segBeg[i], begs - wIncr)
        ends = np.minimum(begs + wLength - 1, segEnd[i])
        eends = np.minimum(ends + wIncr, segEnd[i])
        segmentTable = np.vstack(
            (segmentTable, np.vstack((bbegs, begs, ends, eends)).T)
        )
    return segmentTable


def unravelMask(mask):
    changePoints = np.diff(1 * mask)
    segBeg = np.where(changePoints == 1)[0] + 1
    segEnd = np.where(changePoints == -1)[0]
    if mask[0] == 1:
        segBeg = np.insert(segBeg, 0, 0)
    if mask[-1] == 1:
        segEnd = np.append(segEnd, np.size(mask) - 1)
    nSegs = np.size(segBeg)
    return changePoints, segBeg, segEnd, nSegs


def trainKBM(data, windowLength, windowRate, kbmSize):
    # Calculate number of gaussian components in the whole gaussian pool
    numberOfComponents = int(np.floor((np.size(data, 0) - windowLength) / windowRate))
    # Add new array for storing the mvn objects
    gmPool = []
    likelihoodVector = np.zeros((numberOfComponents, 1))
    muVector = np.zeros((numberOfComponents, np.size(data, 1)))
    sigmaVector = np.zeros((numberOfComponents, np.size(data, 1)))
    for i in range(numberOfComponents):
        mu = np.mean(
            data[np.arange((i * windowRate), (i * windowRate + windowLength), 1, int)],
            axis=0,
        )
        std = np.std(
            data[np.arange((i * windowRate), (i * windowRate + windowLength), 1, int)],
            axis=0,
        )
        muVector[i], sigmaVector[i] = mu, std
        mvn = multivariate_normal(mu, std)
        gmPool.append(mvn)
        likelihoodVector[i] = -np.sum(
            mvn.logpdf(
                data[
                    np.arange((i * windowRate), (i * windowRate + windowLength), 1, int)
                ]
            )
        )
    # Define the global dissimilarity vector
    v_dist = np.inf * np.ones((numberOfComponents, 1))
    # Create the kbm itself, which is a vector of kbmSize size, and contains the gaussian IDs of the components
    kbm = np.zeros((kbmSize, 1))
    # As the stored likelihoods are negative, get the minimum likelihood
    bestGaussianID = np.where(likelihoodVector == np.min(likelihoodVector))[0]
    currentGaussianID = bestGaussianID
    kbm[0] = currentGaussianID
    v_dist[currentGaussianID] = -np.inf
    # Compare the current gaussian with the remaining ones
    dpairsAll = cdist(muVector, muVector, metric="cosine")
    np.fill_diagonal(dpairsAll, -np.inf)
    for j in range(1, kbmSize):
        dpairs = dpairsAll[currentGaussianID]
        v_dist = np.minimum(v_dist, dpairs.T)
        # Once all distances are computed, get the position with highest value
        # set this position to 1 in the binary KBM ponemos a 1 en el vector kbm
        # store the gaussian ID in the KBM
        currentGaussianID = np.where(v_dist == np.max(v_dist))[0]
        kbm[j] = currentGaussianID
        v_dist[currentGaussianID] = -np.inf
    return [kbm, gmPool]


def getVgMatrix(data, gmPool, kbm, topGaussiansPerFrame):

    logLikelihoodTable = getLikelihoodTable(data, gmPool, kbm)

    # The original code was:
    #     Vg = np.argsort(-logLikelihoodTable)[:, 0:topGaussiansPerFrame]
    #     return Vg
    # However this sorts the entire likelihood table, but we only need the top five,
    # thich argpartition does in linear time
    partition_args = np.argpartition(-logLikelihoodTable, 5, axis=1)[:, :5]
    partition = np.take_along_axis(-logLikelihoodTable, partition_args, axis=1)
    vg = np.take_along_axis(partition_args, np.argsort(partition), axis=1)

    return vg


def getLikelihoodTable(data, gmPool, kbm):
    # GETLIKELIHOODTABLE computes the log-likelihood of each feature in DATA
    # against all the Gaussians of GMPOOL specified by KBM vector
    # Inputs:
    #   DATA = matrix of feature vectors
    #   GMPOOL = pool of Gaussians of the kbm model
    #   KBM = vector of the IDs of the actual Gaussians of the KBM
    # Output:
    #   LOGLIKELIHOODTABLE = NxM matrix storing the log-likelihood of each of
    #   the N features given each of th M Gaussians in the KBM
    kbmSize = np.size(kbm, 0)
    logLikelihoodTable = np.zeros([np.size(data, 0), kbmSize])
    for i in range(kbmSize):
        # logging.info("pdf", i, gmPool[int(kbm[i])].logpdf(data).shape)
        logLikelihoodTable[:, i] = gmPool[int(kbm[i])].logpdf(data)
    return logLikelihoodTable


def getSegmentBKs(segmentTable, kbmSize, Vg, bitsPerSegmentFactor, speechMapping):
    # GETSEGMENTBKS converts each of the segments in SEGMENTTABLE into a binary key
    # and/or cumulative vector.

    # Inputs:
    #   SEGMENTTABLE = matrix containing temporal segments returned by 'getSegmentTable' function
    #   KBMSIZE = number of components in the kbm model
    #   VG = matrix of the top components per frame returned by 'getVgMatrix' function
    #   BITSPERSEGMENTFACTOR = proportion of bits that will be set to 1 in the binary keys
    # Output:
    #   SEGMENTBKTABLE = NxKBMSIZE matrix containing N binary keys for each N segments in SEGMENTTABLE
    #   SEGMENTCVTABLE = NxKBMSIZE matrix containing N cumulative vectors for each N segments in SEGMENTTABLE

    numberOfSegments = np.size(segmentTable, 0)
    segmentBKTable = np.zeros([numberOfSegments, kbmSize])
    segmentCVTable = np.zeros([numberOfSegments, kbmSize])
    for i in range(numberOfSegments):
        # Conform the segment according to the segmentTable matrix
        beginningIndex = int(segmentTable[i, 0])
        endIndex = int(segmentTable[i, 3])
        # Store indices of features of the segment
        # speechMapping is substracted one because 1-indexing is used for this variable
        A = np.arange(
            speechMapping[beginningIndex] - 1, speechMapping[endIndex], dtype=int
        )
        segmentBKTable[i], segmentCVTable[i] = binarizeFeatures(
            kbmSize, Vg[A, :], bitsPerSegmentFactor
        )
    # print('done')
    return segmentBKTable, segmentCVTable


def binarizeFeatures(binaryKeySize, topComponentIndicesMatrix, bitsPerSegmentFactor):
    # BINARIZEMATRIX Extracts a binary key and a cumulative vector from the the
    # rows of VG specified by vector A

    # Inputs:
    #   BINARYKEYSIZE = binary key size
    #   TOPCOMPONENTINDICESMATRIX = matrix of top Gaussians per frame
    #   BITSPERSEGMENTFACTOR = Proportion of positions of the binary key which will be set to 1
    # Output:
    #   BINARYKEY = 1xBINARYKEYSIZE binary key
    #   V_F = 1xBINARYKEYSIZE cumulative vector
    numberOfElementsBinaryKey = np.floor(binaryKeySize * bitsPerSegmentFactor)
    # Declare binaryKey
    binaryKey = np.zeros([1, binaryKeySize])
    # Declare cumulative vector v_f
    v_f = np.zeros([1, binaryKeySize])
    unique, counts = np.unique(topComponentIndicesMatrix, return_counts=True)
    # Fill CV
    v_f[:, unique] = counts
    # Fill BK
    binaryKey[0, np.argsort(-v_f)[0][0 : int(numberOfElementsBinaryKey)]] = 1
    # CV normalization
    vf_sum = np.sum(v_f)
    if vf_sum != 0:
        v_f = v_f / vf_sum
    return binaryKey, v_f


def get_sim_mat(X):
    """Returns the similarity matrix based on cosine similarities.
    Arguments
    ---------
    X : array
        (n_samples, n_features).
        Embeddings extracted from the model.
    Returns
    -------
    M : array
        (n_samples, n_samples).
        Similarity matrix with cosine similarities between each pair of embedding.
    """

    # Cosine similarities
    M = sklearn.metrics.pairwise.cosine_similarity(X, X)
    return M


def p_pruning(A, pval):
    n_elems = int((1 - pval) * A.shape[0])

    # For each row in a affinity matrix
    for i in range(A.shape[0]):
        low_indexes = np.argsort(A[i, :])
        low_indexes = low_indexes[0:n_elems]

        # Replace smaller similarity values by 0s
        A[i, low_indexes] = 0

    return A


def compute_affinity_matrix(X):
    """Compute the affinity matrix from data.
    Note that the range of affinity is [0,1].
    Args:
        X: numpy array of shape (n_samples, n_features)
    Returns:
        affinity: numpy array of shape (n_samples, n_samples)
    """
    # Normalize the data.
    l2_norms = np.linalg.norm(X, axis=1)
    X_normalized = X / l2_norms[:, None]
    # Compute cosine similarities. Range is [-1,1].
    cosine_similarities = np.matmul(X_normalized, np.transpose(X_normalized))
    # Compute the affinity. Range is [0,1].
    # Note that this step is not mentioned in the paper!
    affinity = (cosine_similarities + 1.0) / 2.0
    return affinity


def _graph_is_connected(graph):
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _graph_connected_component(graph, node_id):
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=bool)
    nodes_to_explore = np.zeros(n_node, dtype=bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def _set_diag(laplacian, value, norm_laplacian):
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[:: n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            laplacian = laplacian.todia()
        else:
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def spectral_clustering(
    affinity,
    n_clusters=8,
    n_components=None,
    eigen_solver=None,
    random_state=None,
    n_init=10,
    eigen_tol=0.0,
    assign_labels="kmeans",
):
    if assign_labels not in ("kmeans", "discretize"):
        raise ValueError(
            "The 'assign_labels' parameter should be "
            "'kmeans' or 'discretize', but '%s' was given" % assign_labels
        )

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    maps = spectral_embedding(
        affinity,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        drop_first=False,
    )

    if assign_labels == "kmeans":
        kmeans = KMeans(n_clusters, random_state=random_state, n_init=n_init).fit(maps)
        labels = kmeans.labels_
    else:
        labels = discretize(maps, random_state=random_state)

    return labels


def spectral_embedding(
    adjacency,
    n_components=20,
    eigen_solver=None,
    random_state=None,
    eigen_tol=0.0,
    norm_laplacian=True,
    drop_first=True,
):
    adjacency = check_symmetric(adjacency)

    eigen_solver = "arpack"
    norm_laplacian = True
    random_state = check_random_state(random_state)
    n_nodes = adjacency.shape[0]
    if not _graph_is_connected(adjacency):
        warnings.warn(
            "Graph is not fully connected, spectral embedding"
            " may not work as expected."
        )
    laplacian, dd = csgraph_laplacian(
        adjacency, normed=norm_laplacian, return_diag=True
    )
    if (
        eigen_solver == "arpack"
        or eigen_solver != "lobpcg"
        and (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)
    ):
        # print("[INFILE] eigen_solver : ", eigen_solver, "norm_laplacian:", norm_laplacian)
        laplacian = _set_diag(laplacian, 1, norm_laplacian)

        try:
            laplacian *= -1
            v0 = random_state.uniform(-1, 1, laplacian.shape[0])
            lambdas, diffusion_map = eigsh(
                laplacian, k=n_components, sigma=1.0, which="LM", tol=eigen_tol, v0=v0
            )
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                embedding = embedding / dd
        except RuntimeError:
            eigen_solver = "lobpcg"
            laplacian *= -1

    embedding = _deterministic_vector_sign_flip(embedding)
    return embedding[:n_components].T


def compute_sorted_eigenvectors(A):
    """Sort eigenvectors by the real part of eigenvalues.
    Args:
        A: the matrix to perform eigen analysis with shape (M, M)
    Returns:
        w: sorted eigenvalues of shape (M,)
        v: sorted eigenvectors, where v[;, i] corresponds to ith largest
           eigenvalue
    """
    # Eigen decomposition.
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    # Sort from largest to smallest.
    index_array = np.argsort(-eigenvalues)
    # Re-order.
    w = eigenvalues[index_array]
    v = eigenvectors[:, index_array]
    return w, v


EPS = 1e-10


def compute_number_of_clusters(eigenvalues, max_clusters=None, stop_eigenvalue=1e-2):
    """

    Compute number of clusters using EigenGap principle.
    Args:
        eigenvalues: sorted eigenvalues of the affinity matrix
        max_clusters: max number of clusters allowed
        stop_eigenvalue: we do not look at eigen values smaller than this
    Returns:
        number of clusters as an integer
    """

    max_delta = 0
    max_delta_index = 0
    range_end = len(eigenvalues)
    if max_clusters and max_clusters + 1 < range_end:
        range_end = max_clusters + 1
    for i in range(1, range_end):
        if eigenvalues[i - 1] < stop_eigenvalue:
            break
        delta = eigenvalues[i - 1] / eigenvalues[i]
        if delta > max_delta:
            max_delta = delta
            max_delta_index = i
    return max_delta_index


def diagonal_fill(A):
    """
    Sets the diagonal elemnts of the matrix to the max of each row
    """
    np.fill_diagonal(A, 0.0)
    A[np.diag_indices(A.shape[0])] = np.max(A, axis=1)
    return A


def gaussian_blur(A, sigma=1.0):
    """
    Does a gaussian blur on the affinity matrix
    """
    return gaussian_filter(A, sigma=sigma)


def row_threshold_mult(A, p=0.95, mult=0.01):
    """
    For each row multiply elements smaller than the row's p'th percentile by mult
    """
    percentiles = np.percentile(A, p * 100, axis=1)
    mask = A < percentiles[:, np.newaxis]

    A = (mask * mult * A) + (~mask * A)
    return A


def row_max_norm(A):
    """
    Row-wise max normalization: S_{ij} = Y_{ij} / max_k(Y_{ik})
    """
    maxes = np.amax(A, axis=1)
    return A / maxes


def sim_enhancement(A):
    func_order = [gaussian_blur, diagonal_fill, row_threshold_mult, row_max_norm]
    for f in func_order:
        A = f(A)
    return A


def binaryKeySimilarity_cdist(clusteringMetric, bkT1, cvT1, bkT2, cvT2):
    if clusteringMetric == "cosine":
        S = 1 - cdist(cvT1, cvT2, metric=clusteringMetric)
    elif clusteringMetric == "jaccard":
        S = 1 - cdist(bkT1, bkT2, metric=clusteringMetric)
    else:
        logging.info("Clustering metric must be cosine or jaccard")
    return S


def getSpectralClustering(
    bestClusteringMetric,
    N_init,
    bkT,
    cvT,
    speaker_count,
    sigma,
    percentile,
    maxNrSpeakers,
    random_state = None,
):
    if speaker_count is None:
        #  Compute affinity matrix.
        simMatrix = binaryKeySimilarity_cdist(bestClusteringMetric, bkT, cvT, bkT, cvT)

        # Laplacian calculation
        affinity = sim_enhancement(simMatrix)

        (eigenvalues, eigenvectors) = compute_sorted_eigenvectors(affinity)
        # Get number of clusters.
        speaker_count = compute_number_of_clusters(eigenvalues, maxNrSpeakers, 1e-2)

    else:
        #  Compute affinity matrix.
        simMatrix = binaryKeySimilarity_cdist(bestClusteringMetric, bkT, cvT, bkT, cvT)

        # Laplacian calculation
        affinity = sim_enhancement(simMatrix)

    bestClusteringID = spectral_clustering(
        affinity,
        n_clusters=speaker_count,
        eigen_solver=None,
        random_state=random_state,
        n_init=25,
        eigen_tol=0.0,
        assign_labels="kmeans",
    )

    return bestClusteringID


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    # From https://stackoverflow.com/a/40443565
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), "valid") / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[: WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def performResegmentation(
    data,
    speechMapping,
    mask,
    finalClusteringTable,
    segmentTable,
    modelSize,
    nbIter,
    smoothWin,
    numberOfSpeechFeatures,
):
    from sklearn import mixture

    np.random.seed(0)

    changePoints, segBeg, segEnd, nSegs = unravelMask(mask)
    speakerIDs = np.unique(finalClusteringTable)
    trainingData = np.empty([2, 0])
    for i in np.arange(np.size(speakerIDs, 0)):
        spkID = speakerIDs[i]
        speakerFeaturesIndxs = []
        idxs = np.where(finalClusteringTable == spkID)[0]
        for l in np.arange(np.size(idxs, 0)):
            speakerFeaturesIndxs = np.append(
                speakerFeaturesIndxs,
                np.arange(
                    int(segmentTable[idxs][:][l, 1]),
                    int(segmentTable[idxs][:][l, 2]) + 1,
                ),
            )
        formattedData = np.vstack(
            (
                np.tile(spkID, (1, np.size(speakerFeaturesIndxs, 0))),
                speakerFeaturesIndxs,
            )
        )
        trainingData = np.hstack((trainingData, formattedData))

    llkMatrix = np.zeros([np.size(speakerIDs, 0), numberOfSpeechFeatures])
    for i in np.arange(np.size(speakerIDs, 0)):
        spkIdxs = np.where(trainingData[0, :] == speakerIDs[i])[0]
        spkIdxs = speechMapping[trainingData[1, spkIdxs].astype(int)].astype(int) - 1
        msize = np.minimum(modelSize, np.size(spkIdxs, 0))
        w_init = np.ones([msize]) / msize
        m_init = data[
            spkIdxs[np.random.randint(np.size(spkIdxs, 0), size=(1, msize))[0]], :
        ]
        gmm = mixture.GaussianMixture(
            n_components=msize,
            covariance_type="diag",
            weights_init=w_init,
            means_init=m_init,
            verbose=0,
        )
        gmm.fit(data[spkIdxs, :])
        llkSpk = gmm.score_samples(data)
        llkSpkSmoothed = np.zeros([1, numberOfSpeechFeatures])
        for jx in np.arange(nSegs):
            sectionIdx = np.arange(
                speechMapping[segBeg[jx]] - 1, speechMapping[segEnd[jx]]
            ).astype(int)
            sectionWin = np.minimum(smoothWin, np.size(sectionIdx))
            if sectionWin % 2 == 0:
                sectionWin = sectionWin - 1
            if sectionWin >= 2:
                llkSpkSmoothed[0, sectionIdx] = smooth(llkSpk[sectionIdx], sectionWin)
            else:
                llkSpkSmoothed[0, sectionIdx] = llkSpk[sectionIdx]
        llkMatrix[i, :] = llkSpkSmoothed[0].T
    segOut = np.argmax(llkMatrix, axis=0) + 1
    segChangePoints = np.diff(segOut)
    changes = np.where(segChangePoints != 0)[0]
    relSegEnds = speechMapping[segEnd]
    relSegEnds = relSegEnds[0:-1]
    changes = np.sort(np.unique(np.hstack((changes, relSegEnds))))

    # Create the new segment and clustering tables
    currentPoint = 0
    finalSegmentTable = np.empty([0, 4])
    finalClusteringTableResegmentation = np.empty([0, 1])

    for i in np.arange(np.size(changes, 0)):
        addedRow = np.hstack(
            (
                np.tile(
                    np.where(speechMapping == np.maximum(currentPoint, 1))[0], (1, 2)
                ),
                np.tile(
                    np.where(speechMapping == np.maximum(1, changes[i].astype(int)))[0],
                    (1, 2),
                ),
            )
        )
        finalSegmentTable = np.vstack((finalSegmentTable, addedRow[0]))
        finalClusteringTableResegmentation = np.vstack(
            (finalClusteringTableResegmentation, segOut[(changes[i]).astype(int)])
        )
        currentPoint = changes[i] + 1
    addedRow = np.hstack(
        (
            np.tile(np.where(speechMapping == currentPoint)[0], (1, 2)),
            np.tile(np.where(speechMapping == numberOfSpeechFeatures)[0], (1, 2)),
        )
    )
    finalSegmentTable = np.vstack((finalSegmentTable, addedRow[0]))
    finalClusteringTableResegmentation = np.vstack(
        (finalClusteringTableResegmentation, segOut[(changes[i] + 1).astype(int)])
    )
    return finalClusteringTableResegmentation, finalSegmentTable
