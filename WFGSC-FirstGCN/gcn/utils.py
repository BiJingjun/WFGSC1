import numpy as np
import scipy.io as sio
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import tensorflow as tf
import numpy as np
import sys
flags = tf.app.flags
FLAGS = flags.FLAGS

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def load_data(dataset_str,path="../data/"):
    path = path + "handwritten/xin/"
    if dataset_str == "cora":
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    elif dataset_str == "citeseer":
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_test = sio.loadmat(path + "test.mat")
        idx_test = idx_test['array'].flatten()
        idx_train = range(120)
        idx_val = range(120, 620)
    elif dataset_str == "pubmed":
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_test = sio.loadmat(path + "test.mat")
        idx_test = idx_test['matrix']
        idx_train = range(60)
        idx_val = range(200, 500)
    elif dataset_str == "scence":
        features = sio.loadmat(path + "feat")
        features = features['feat']
        adj = sio.loadmat(path + "adj")
        adj = adj['adj']
        labels = sio.loadmat(path + "label")
        labels = labels['label']
        idx_train = range(250+FLAGS.pre*250)
        idx_val = range(250+FLAGS.pre*250, 750+FLAGS.pre*250)
        idx_test = range(750+FLAGS.pre*250, 4485)
    elif dataset_str == "largecora":
        data = sio.loadmat(path + "large_cora")
        l = data['labels'].flatten()
        labels = np.zeros([l.shape[0], np.max(l) + 1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
        features = data['X']
        adj = data['G']
    else:
        features = sio.loadmat(path + "hand" + "feat1")
        features = features['feat1']

        labels = sio.loadmat(path + "label")
        labels = labels['label']
        if labels.min() == 1:
            labels = labels - 1

    trainmask = sio.loadmat(path + "hand" + "train_idx0" + dataset_str)
    idx_train = trainmask['train_idx0']
    valmask = sio.loadmat(path + "hand" + "valid_idx0" + dataset_str)
    idx_val = valmask['valid_idx0']
    testmask = sio.loadmat(path + "hand" + "test_idx0" + dataset_str)
    idx_test = testmask['test_idx0']

    label = np.zeros((labels.shape[0], labels.max() + 1))
    for i in range(labels.shape[0]):
        label[i][labels[i]] = 1
    labels = label


    x1 = features.todense().A

    features1 = sio.loadmat(path + "hand1feat1")
    features1 = features1['feat1']
    features1 = features1.toarray()
    sname1, Fout1 = MLAN_FME_V3_0_1(features1, labels, idx_train)
    sname1 = np.nan_to_num(sname1, nan=1e-20)
    alpha_1 = generate_alpha_v(sname1, features1)



    features2 = sio.loadmat(path + "hand2feat1")
    features2 = features2['feat1']
    features2 = features2.toarray()
    sname2, Fout2 = MLAN_FME_V3_0_1(features2, labels, idx_train)
    sname2 = np.nan_to_num(sname2, nan=1e-20)
    alpha_2 = generate_alpha_v(sname2, features2)


    features3 = sio.loadmat(path + "hand3feat1")
    features3 = features3['feat1']
    features3 = features3.toarray()
    sname3, Fout3 = MLAN_FME_V3_0_1(features3, labels, idx_train)
    sname3 = np.nan_to_num(sname3, nan=1e-20)
    alpha_3 = generate_alpha_v(sname3, features3)


    features4 = sio.loadmat(path + "hand4feat1")
    features4 = features4['feat1']
    features4 = features4.toarray()
    sname4, Fout4 = MLAN_FME_V3_0_1(features4, labels, idx_train)
    sname4 = np.nan_to_num(sname4, nan=1e-20)
    alpha_4 = generate_alpha_v(sname4, features4)


    features5 = sio.loadmat(path + "hand5feat1")
    features5 = features5['feat1']
    features5 = features5.toarray()
    sname5, Fout5 = MLAN_FME_V3_0_1(features5, labels, idx_train)
    sname5 = np.nan_to_num(sname5, nan=1e-20)
    alpha_5 = generate_alpha_v(sname5, features5)


    features6 = sio.loadmat(path + "hand6feat1")
    features6 = features6['feat1']
    features6 = features6.toarray()
    sname6, Fout6 = MLAN_FME_V3_0_1(features6, labels, idx_train)
    sname6 = np.nan_to_num(sname6, nan=1e-20)
    alpha_6 = generate_alpha_v(sname6, features6)


    alpha_1 = 1 / alpha_1
    alpha_2 = 1 / alpha_2
    alpha_3 = 1 / alpha_3
    alpha_4 = 1 / alpha_4
    alpha_5 = 1 / alpha_5
    alpha_6 = 1 / alpha_6
    alphasum = alpha_1 + alpha_2 + alpha_3 + alpha_4 + alpha_5 + alpha_6

    adj = (alpha_1 * sname1 + alpha_2 * sname2 + alpha_3 * sname3 + alpha_4 * sname4 + alpha_5 * sname5+ alpha_6 * sname6)/alphasum

    adj = sp.csr_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # sio.savemat("handSout", {'adj': adj})

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    Fcon = sp.csc_matrix(np.concatenate((features1, features2, features3, features4, features5, features6), axis=1))

    Fcon_filled = np.nan_to_num(Fcon, nan=1e-20)


    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

##########################################################################
def preprocess_laplacian(adj):
    """Preprocessing of laplacian"""
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    #coo = laplacian.tocoo().astype(np.float32)
    #indices = np.mat([coo.row, coo.col]).transpose()
    #return tf.SparseTensor(indices, coo.data, coo.shape)

    return laplacian

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def MLAN_FME_V3_0_1(X,samp_label,labeld_data):
    num_labeled_data = len(labeld_data)
    lambda0 = 5.0
    k=5
    para_lamda = 0.003
    para_mu = 0
    c = samp_label.shape[1]

    num = len(X)
    labeled_N = num_labeled_data

    List = labeld_data.T
    Liall = range(0,num)
    List_ = np.setdiff1d(Liall, List).T
    groundtruth = np.zeros((num, c), dtype='float32')
    groundtruth[List,:] = samp_label[List,:]
    groundtruth[List_,:] = samp_label[List_,:]
    Y = np.zeros((num, c), dtype='float32')
    Y[List,:] = samp_label[List,:]
    Y[List_,:] = 0
    Y_L = groundtruth[List,:][0]

    for j in range(0,num):
        X[j,:] = [X[j,:] - np.mean(X[j,:] ) ] / np.std(X[j,:] )

    SUM = np.zeros((num,num), dtype='float32')
    distX_initial =  L2_distance_1(X.T,X.T)
    SUM = SUM + distX_initial

    distX = SUM
    distXs = np.zeros((num,num), dtype='float32')
    idx = np.zeros((num, num), dtype='int')
    for j in range(0, num):
        distXs[j,:] = np.sort(distX[j,:])
        idx[j,:] = np.argsort(distX[j,:])

    S = np.zeros((num,num), dtype='float32')
    rr = np.zeros((num,1), dtype='float32')

    for i in range(0, num):
        di = distXs[i, 1:k + 2]
        rr[i] = 0.5 * (k * di[k] - sum(di[0:k]))
        id = idx[i, 1:k + 2]
        dikdi = di[k] - di
        ksum = k * di[k] - sum(di[0:k])
        Siid = dikdi/ksum
        nanidx = np.isnan(Siid)
        Siid[nanidx] = 1e-20
        S[i, id] = Siid

    alpha = np.mean(rr)
    para_uu = 0
    para_ul = 1
    X_ = X.T

    S_old = np.zeros((num,num), dtype='float32')
    p = 2
    p2 = 1 / (p - 1)
    SUM = np.zeros((num,num), dtype='float32')
    S = (S + S.T)/2
    sunS=np.sum(S,axis=1)
    L = np.diag(sunS) -S

    t0 = np.trace(np.dot(np.dot(X.T, L),X) )
    trace_X = (np.trace(np.dot(np.dot(X.T, L),X) )) ** p2
    trace_XX = 1/trace_X
    sum_trace_X = trace_XX
    Wv = 1 / (trace_X * sum_trace_X)
    distX_updated = (Wv ** p) * distX_initial

    SUM = SUM + distX_updated
    #weight_matrix= Wv

    distX = SUM
    W,b,F = FME_semi(X_, L, Y ,para_lamda,para_uu,para_ul)
    F[List,:]=groundtruth[List,:]

    distf = L2_distance_1(F.T,F.T)
    S = np.zeros((num,num), dtype='float32')
    for i in range(0, num):
        idxa0 = idx[i, 1:k + 1]
        dfi = distf[i, idxa0]
        dxi = distX[i, idxa0]
        ad = -(dxi +lambda0 * dfi) / (2 * alpha)
        #ad_matrix[i,:]=[ad]
        #Si = EProjSimplex_new(ad)
        S[i, idxa0] = EProjSimplex_new(ad)

    S = (S + S.T)/2


    return S,F


def L2_distance_1(a,b):
    aa0 = a*a
    aa = np.sum(aa0,0).reshape(-1,1)
    bb0 = b * b
    bb = np.sum(bb0,0).reshape(1,-1)
    ab=np.dot(a.T,b)
    repmataa = np.tile(aa,[1,len(aa)])
    repmatbb = np.tile(bb, [len(aa), 1])
    #repmataa = np.matlib.repmat(aa.T, 1, np.shape(bb)[1])
    d = repmataa + repmatbb - 2*ab
    #d = real(d);
    d = np.maximum(d,0)

    return d

def FME_semi(X,L,T,para_lamda,para_uu,para_ul):
    dim = X.shape[0]
    n = X.shape[1]
    sumT = np.sum(T,1)

    labeled_idx = sumT == 1
    Xm = np.mean(X, 1)
    npones = np.ones((1, n), dtype='float32')
    Xm = Xm.reshape(dim,1)
    npdot = np.dot(Xm, npones)
    Xc = X - npdot
    if dim < n:
        St = np.dot(Xc, Xc.T)
        A = para_lamda * np.dot(np.linalg.inv(para_lamda * St + np.eye(dim)), Xc)
    else:
        K= np.dot(Xc.T, Xc)
        A = para_lamda * np.dot(Xc,np.linalg.inv(para_lamda * K + np.eye(n)))

    u = para_uu * np.ones((n,1), dtype='float32')
    u[labeled_idx] = para_ul
    U = np.diag(u[:,0])

    M = U + L

    F = np.dot(np.linalg.inv(M) , np.dot(U,T))
    W = np.dot(A , F)

    bb2 = np.dot(np.dot(W.T , X) , np.ones((n,1), dtype='float32'))
    bb1 = np.sum(F, 0).T.reshape(-1,1)
    b = 1 / n * (bb1 - bb2)

    return W,b,F

def EProjSimplex_new(v):
    k=1

    ft = 1
    n = len(v)
    v0 = v - np.mean(v) + k / n

    vmin = np.min(v0)

    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 10 ** -10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f / g
            ft = ft + 1
            if ft > 100:
                x = np.max(v1, 0)
                break
        x = np.maximum(v1, 0)

    else:
        x=v0

    return x


def generate_alpha_v(sname1,features1):
    sname1 = (sname1 + sname1.T) / 2
    sunS = np.sum(sname1, axis=1)
    L_v = np.diag(sunS) - sname1
    L_v_filled = np.nan_to_num(L_v, nan=1e-20)
    alpha_v = 1 / (2 * np.sqrt((np.trace(np.dot(np.dot(features1.T, L_v_filled), features1)))))

    return alpha_v