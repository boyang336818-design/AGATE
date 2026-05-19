from __future__ import division
from __future__ import print_function
import os, sys
#from tempfile import TemporaryFile

import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops
from torch_geometric.data import Data, DataLoader

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_auc_score, average_precision_score
import random

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 42
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
np.random.seed(SEED)
torch.manual_seed(SEED)
from torch import optim
import torch.nn.functional as F
from gae.model import GCNModelVAE, GCNModelAE, GCNModelAES, augment_graph, GAE,DGAE
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from deepWalk.graph import load_edgelist_from_csr_matrix, build_deepwalk_corpus_iter, build_deepwalk_corpus
from deepWalk.skipGram import SkipGram
from sklearn.cluster import KMeans
# from gae.clustering_metric import clustering_metrics
from gae.clustering_metric import *
from tqdm import tqdm
import matplotlib.pyplot as plt
# from scipy.sparse.csgraph import connected_components
# from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['NSimSun']
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_ae', help="models used")
parser.add_argument('--dw', type=int, default=1, help="whether to use deepWalk regularization, 0/1")
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=128, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--walk-length', default=5, type=int, help='Length of the random walk started at each node')
parser.add_argument('--window-size', default=3, type=int, help='Window size of skipgram model.')
parser.add_argument('--number-walks', default=5, type=int, help='Number of random walks to start at each node')#每个节点开始走多少次
parser.add_argument('--full-number-walks', default=0, type=int, help='Number of random walks from each node')#每个节点走多少次
parser.add_argument('--lr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')
parser.add_argument('--context', type=int, default=1, help="whether to use context nodes for skipgram")
parser.add_argument('--ns', type=int, default=0, help="whether to use negative samples for skipgram")
parser.add_argument('--n-clusters', default=7, type=int, help='number of clusters, 7 for cora, 6 for citeseer')
parser.add_argument('--plot', type=int, default=0, help="whether to plot the clusters using tsne")
args = parser.parse_args()

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.neighbors import KDTree
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import normalize





########################################################################
# TORC CLUSTERING
########################################################################
class TORC:

    """
    Stable TORC clustering
    Fixed version
    """

    def __init__(
            self,
            metric='cosine',
            k=5,
            cut_ratio=0.02,
            min_cluster_size=3
    ):

        self.metric = metric
        self.k = k
        self.cut_ratio = cut_ratio
        self.min_cluster_size = min_cluster_size

    ####################################################################
    # FIT
    ####################################################################

    def fit_predict(self, X):

        from sklearn.preprocessing import normalize

        ############################################################
        # normalize embedding
        ############################################################

        if self.metric == 'cosine':
            X = normalize(X)

        n = X.shape[0]

        ############################################################
        # pairwise distance
        ############################################################

        D = pairwise_distances(
            X,
            metric=self.metric
        )

        ############################################################
        # build KNN graph
        ############################################################

        knn_graph = np.zeros((n, n))

        for i in range(n):

            nn_ids = np.argsort(D[i])[1:self.k + 1]

            for j in nn_ids:

                knn_graph[i, j] = D[i, j]
                knn_graph[j, i] = D[i, j]

        ############################################################
        # build MST
        ############################################################

        mst = minimum_spanning_tree(
            csr_matrix(knn_graph)
        ).toarray()

        rows, cols = np.nonzero(mst)

        edges = []

        ############################################################
        # compute torque
        ############################################################

        for u, v in zip(rows, cols):

            dist = mst[u, v]

            ########################################################
            # local density
            ########################################################

            mass_u = np.sum(
                np.exp(
                    -knn_graph[u][knn_graph[u] > 0]
                )
            )

            mass_v = np.sum(
                np.exp(
                    -knn_graph[v][knn_graph[v] > 0]
                )
            )

            ########################################################
            # torque
            ########################################################

            torque = (
                    mass_u *
                    mass_v *
                    dist *
                    dist
            )

            edges.append(
                (
                    u,
                    v,
                    dist,
                    torque
                )
            )

        ############################################################
        # no edge case
        ############################################################

        if len(edges) == 0:

            return np.zeros(n, dtype=int)

        ############################################################
        # sort torque descending
        ############################################################

        edges.sort(
            key=lambda x: x[3],
            reverse=True
        )

        torques = np.array([
            e[3] for e in edges
        ])

        ############################################################
        # single edge case
        ############################################################

        if len(torques) == 1:

            return np.zeros(n, dtype=int)

        ############################################################
        # torque gap
        ############################################################

        ratios = []

        for i in range(len(torques) - 1):

            if torques[i + 1] <= 1e-12:

                ratios.append(0)

            else:

                ratios.append(
                    torques[i]
                    /
                    torques[i + 1]
                )

        ############################################################
        # adaptive cut
        ############################################################

        cut_idx = np.argmax(ratios)

        ############################################################
        # prevent over cutting
        ############################################################

        max_cut = max(
            1,
            int(
                self.cut_ratio * len(edges)
            )
        )

        cut_idx = min(
            cut_idx,
            max_cut
        )

        abnormal_edges = edges[:cut_idx + 1]

        ############################################################
        # IMPORTANT:
        # USE MST
        ############################################################

        G = (mst > 0).astype(np.uint8)

        ############################################################
        # remove abnormal edges
        ############################################################

        for u, v, _, _ in abnormal_edges:

            G[u, v] = 0
            G[v, u] = 0

        ############################################################
        # connected components
        ############################################################

        n_comp, labels = connected_components(
            csr_matrix(G),
            directed=False
        )

        print(f"TORC discovered {n_comp} clusters")

        ############################################################
        # merge tiny clusters
        ############################################################

        unique, counts = np.unique(
            labels,
            return_counts=True
        )

        small_clusters = unique[
            counts < self.min_cluster_size
        ]

        major_clusters = unique[
            counts >= self.min_cluster_size
        ]

        ############################################################
        # merge small clusters
        ############################################################

        if len(small_clusters) > 0 and len(major_clusters) > 0:

            major_centers = []

            for mc in major_clusters:

                idx = np.where(labels == mc)[0]

                center = np.mean(
                    X[idx],
                    axis=0
                )

                major_centers.append(center)

            major_centers = np.array(
                major_centers
            )

            for sc in small_clusters:

                idx = np.where(labels == sc)[0]

                for node in idx:

                    d = pairwise_distances(
                        X[node].reshape(1, -1),
                        major_centers,
                        metric=self.metric
                    )[0]

                    best_cluster = major_clusters[
                        np.argmin(d)
                    ]

                    labels[node] = best_cluster

        ############################################################
        # relabel
        ############################################################

        unique_labels = np.unique(labels)

        label_map = {
            old: new
            for new, old
            in enumerate(unique_labels)
        }

        labels = np.array([
            label_map[l]
            for l in labels
        ])

        return labels
########################################################################
# IMPROVED TORC
########################################################################

class TORC2:

    """
    Improved Stable TORC Clustering

    Key fixes:
    1. Use KNN graph as final graph (NOT MST)
    2. Mean density instead of sum density
    3. torque = density_u * density_v * dist
       (NOT dist^2)
    4. Statistical threshold instead of ratio-gap
    5. Symmetric KNN graph
    """

    def __init__(
            self,
            metric='cosine',
            k=15,
            cut_std=2.0,
            min_cluster_size=10
    ):

        self.metric = metric
        self.k = k
        self.cut_std = cut_std
        self.min_cluster_size = min_cluster_size

    ####################################################################
    # FIT
    ####################################################################

    def fit_predict(self, X):

        from sklearn.preprocessing import normalize

        ############################################################
        # normalize embedding
        ############################################################

        if self.metric == 'cosine':
            X = normalize(X)

        n = X.shape[0]

        ############################################################
        # pairwise distance
        ############################################################

        D = pairwise_distances(
            X,
            metric=self.metric
        )

        ############################################################
        # build symmetric KNN graph
        ############################################################

        knn_graph = np.zeros((n, n))

        for i in range(n):

            nn_ids = np.argsort(D[i])[1:self.k + 1]

            for j in nn_ids:

                knn_graph[i, j] = D[i, j]
                knn_graph[j, i] = D[i, j]

        ############################################################
        # MST only for abnormal edge discovery
        ############################################################

        mst = minimum_spanning_tree(
            csr_matrix(knn_graph)
        ).toarray()

        rows, cols = np.nonzero(mst)

        edges = []

        ############################################################
        # compute torque
        ############################################################

        for u, v in zip(rows, cols):

            dist = mst[u, v]

            ########################################################
            # local density (MEAN not SUM)
            ########################################################

            nbr_u = knn_graph[u][knn_graph[u] > 0]
            nbr_v = knn_graph[v][knn_graph[v] > 0]

            if len(nbr_u) == 0 or len(nbr_v) == 0:
                continue

            density_u = np.mean(
                np.exp(-nbr_u)
            )

            density_v = np.mean(
                np.exp(-nbr_v)
            )

            ########################################################
            # improved torque
            ########################################################

            torque = (
                    density_u *
                    density_v *
                    dist
            )

            edges.append(
                (
                    u,
                    v,
                    dist,
                    torque
                )
            )

        ############################################################
        # edge empty
        ############################################################

        if len(edges) == 0:

            return np.zeros(n, dtype=int)

        ############################################################
        # torques
        ############################################################

        torques = np.array([
            e[3] for e in edges
        ])

        ############################################################
        # statistical threshold
        ############################################################

        mean_tau = np.mean(torques)
        std_tau = np.std(torques)

        threshold = (
                mean_tau +
                self.cut_std * std_tau
        )

        ############################################################
        # abnormal edges
        ############################################################

        abnormal_edges = []

        for e in edges:

            if e[3] > threshold:
                abnormal_edges.append(e)

        ############################################################
        # IMPORTANT:
        # USE KNN GRAPH
        # NOT MST
        ############################################################

        G = (knn_graph > 0).astype(np.uint8)

        ############################################################
        # remove abnormal edges
        ############################################################

        for u, v, _, _ in abnormal_edges:

            G[u, v] = 0
            G[v, u] = 0

        ############################################################
        # connected components
        ############################################################

        n_comp, labels = connected_components(
            csr_matrix(G),
            directed=False
        )

        print(f"TORC discovered {n_comp} clusters")

        ############################################################
        # merge tiny clusters
        ############################################################

        unique, counts = np.unique(
            labels,
            return_counts=True
        )

        small_clusters = unique[
            counts < self.min_cluster_size
        ]

        major_clusters = unique[
            counts >= self.min_cluster_size
        ]

        ############################################################
        # merge tiny clusters
        ############################################################

        if len(small_clusters) > 0 and len(major_clusters) > 0:

            major_centers = []

            for mc in major_clusters:

                idx = np.where(labels == mc)[0]

                center = np.mean(
                    X[idx],
                    axis=0
                )

                major_centers.append(center)

            major_centers = np.array(
                major_centers
            )

            ########################################################
            # assign small cluster nodes
            ########################################################

            for sc in small_clusters:

                idx = np.where(labels == sc)[0]

                for node in idx:

                    d = pairwise_distances(
                        X[node].reshape(1, -1),
                        major_centers,
                        metric=self.metric
                    )[0]

                    best_cluster = major_clusters[
                        np.argmin(d)
                    ]

                    labels[node] = best_cluster

        ############################################################
        # relabel
        ############################################################

        unique_labels = np.unique(labels)

        label_map = {
            old: new
            for new, old
            in enumerate(unique_labels)
        }

        labels = np.array([
            label_map[l]
            for l in labels
        ])

        return labels
class TORC1:
    """
    Torque Clustering
    Autonomous clustering by fast find of mass and distance peaks
    """

    def __init__(self, metric='euclidean'):
        self.metric = metric

    def fit_predict(self, X):

        self.X = X
        self.n_samples = len(X)

        self.D = pairwise_distances(
            X,
            metric=self.metric
        )

        all_connections, final_graph = self._build_hierarchy()

        torques = np.array([
            c['mass_product'] * c['distance_sq']
            for c in all_connections
        ])

        if len(torques) == 0:
            return np.zeros(self.n_samples, dtype=int)

        order = np.argsort(-torques)

        TSCL = [all_connections[i] for i in order]
        sorted_tau = torques[order]

        abnormal_edges = self._find_abnormal_connections(
            TSCL,
            sorted_tau
        )

        G = final_graph.copy()

        for edge in abnormal_edges:
            u, v = edge
            G[u, v] = 0
            G[v, u] = 0

        n_components, labels = connected_components(
            csr_matrix(G),
            directed=False
        )

        return labels

    ####################################################################
    # hierarchy evolution
    ####################################################################

    def _build_hierarchy(self):

        n = self.n_samples

        clusters = [{i} for i in range(n)]

        global_graph = np.zeros((n, n), dtype=np.uint8)

        all_connections = []

        while len(clusters) > 1:

            masses = np.array([len(c) for c in clusters])

            edges = []

            for i, ci in enumerate(clusters):

                best_j = None
                best_dist = np.inf

                for j, cj in enumerate(clusters):

                    if i == j:
                        continue

                    if masses[j] < masses[i]:
                        continue

                    d = self._cluster_distance(ci, cj)

                    if d < best_dist:
                        best_dist = d
                        best_j = j

                if best_j is not None:

                    cj = clusters[best_j]

                    u, v, real_dist = self._closest_points(
                        ci,
                        cj
                    )

                    edges.append(
                        (i, best_j, u, v, real_dist)
                    )

            if len(edges) == 0:
                break

            cluster_adj = np.zeros(
                (len(clusters), len(clusters)),
                dtype=np.uint8
            )

            for i, j, u, v, dist in edges:

                cluster_adj[i, j] = 1
                cluster_adj[j, i] = 1

                global_graph[u, v] = 1
                global_graph[v, u] = 1

                Mi = masses[i] * masses[j]
                Di = dist ** 2

                all_connections.append({
                    'edge': (u, v),
                    'mass_product': Mi,
                    'distance_sq': Di,
                    'distance': dist
                })

            n_comp, labels = connected_components(
                csr_matrix(cluster_adj),
                directed=False
            )

            new_clusters = []

            for k in range(n_comp):

                merged = set()

                idxs = np.where(labels == k)[0]

                for idx in idxs:
                    merged.update(clusters[idx])

                new_clusters.append(merged)

            if len(new_clusters) == len(clusters):
                break

            clusters = new_clusters

        return all_connections, global_graph

    ####################################################################
    # abnormal connections
    ####################################################################

    def _find_abnormal_connections(
            self,
            TSCL,
            sorted_tau
    ):

        if len(sorted_tau) <= 1:
            return []

        M_all = np.array([
            c['mass_product']
            for c in TSCL
        ])

        D_all = np.array([
            c['distance_sq']
            for c in TSCL
        ])

        mean_tau = np.mean(sorted_tau)
        mean_M = np.mean(M_all)
        mean_D = np.mean(D_all)

        large_mask = (
                (sorted_tau >= mean_tau) &
                (M_all >= mean_M) &
                (D_all >= mean_D)
        )

        large_count = np.sum(large_mask)

        if large_count == 0:
            return []

        tgaps = []

        for i in range(len(sorted_tau) - 1):

            if sorted_tau[i + 1] == 0:
                tgaps.append(0)
                continue

            top_i_large = np.sum(
                large_mask[:i + 1]
            )

            omega_i = (
                    top_i_large / large_count
            )

            tgap = omega_i * (
                    sorted_tau[i]
                    / sorted_tau[i + 1]
            )

            tgaps.append(tgap)

        L = np.argmax(tgaps) + 1

        abnormal_connections = []

        for i in range(L):
            abnormal_connections.append(
                TSCL[i]['edge']
            )

        return abnormal_connections

    ####################################################################
    # cluster distance
    ####################################################################

    def _cluster_distance(self, c1, c2):

        c1 = list(c1)
        c2 = list(c2)

        sub = self.D[np.ix_(c1, c2)]

        return np.min(sub)

    ####################################################################
    # closest points
    ####################################################################

    def _closest_points(self, c1, c2):

        c1 = list(c1)
        c2 = list(c2)

        sub = self.D[np.ix_(c1, c2)]

        idx = np.argmin(sub)

        i, j = np.unravel_index(
            idx,
            sub.shape
        )

        u = c1[i]
        v = c2[j]

        return u, v, sub[i, j]


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, y_test, tx, ty, test_maks, true_labels = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape
    #nclass = args.class_num


    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    edges = train_edges



    # Before proceeding further, make the structure for doing deepWalk
    if args.dw == 1:
        print('Using deepWalk regularization...')
        G = load_edgelist_from_csr_matrix(adj_orig, undirected=True)
        print("Number of nodes: {}".format(len(G.nodes())))
        num_walks = len(G.nodes()) * args.number_walks       #从所有节点走，每个节点走number_walks次，那总的行走的次数总结点数*number_walks
        print("Number of walks: {}".format(num_walks))
        data_size = num_walks * args.walk_length            # 行走的总数据大小
        print("Data size (walks*length): {}".format(data_size))

    # Some preprocessing
    # randomlist = []
    # arraylen = adj.shape[0]
    # while len(randomlist) < (arraylen / 30):
    #     random_int = np.random.randint(0, arraylen)
    #     if random_int not in randomlist:
    #         randomlist.append(random_int)
    #     for position in randomlist:
    #         adj[position] = 0  # 29%
    # adj = adj
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    if args.model == 'gcn_ae':
        model = GCNModelAE(feat_dim, args.hidden1, args.hidden2,args.dropout)
        # model = GAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    else:
        # model = GAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # log_sigma_dw = nn.Parameter(torch.tensor(0.0))
    if args.dw == 1:

        sg = SkipGram(args.hidden2, adj.shape[0])
        optimizer_dw = optim.Adam(sg.parameters(), lr=args.lr_dw)
        # optimizer_dw = optim.Adam(list(sg.parameters()) + [log_sigma_dw], lr=args.lr_dw)


        # Construct the nodes for doing random walk. Doing it before since the seed is fixed
        nodes_in_G = list(G.nodes())
        chunks = len(nodes_in_G) // args.number_walks
        random.Random().shuffle(nodes_in_G)

    hidden_emb = None


    # 初始化最佳聚类指标变量
    best_acc = 0.0
    best_nmi = 0.0
    best_ari = 0.0
    best_cluster_epoch = 0
    best_hidden_emb = None

    for epoch in tqdm(range(args.epochs)):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        z, mu, logvar = model(features, adj_norm)
        # After back-propagating gae loss, now do the deepWalk regularization
        if args.dw == 1:
            sg.train()
            if args.full_number_walks > 0:
                walks = build_deepwalk_corpus(G, num_paths=args.full_number_walks,
                                              path_length=args.walk_length, alpha=0,
                                              rand=random.Random(SEED))
            else:
                walks = build_deepwalk_corpus_iter(G, num_paths=args.number_walks,
                                                   path_length=args.walk_length, alpha=0,
                                                   rand=random.Random(SEED),
                                                   chunk=epoch % chunks,
                                                   nodes=nodes_in_G)
            for walk in walks:
                if args.context == 1:

                    for center_node_pos in range(len(walk)):

                        curr_pair = (
                            int(walk[center_node_pos]),
                            []
                        )

                        for w in range(
                                -args.window_size,
                                args.window_size + 1
                        ):

                            context_node_pos = (
                                    center_node_pos + w
                            )

                            if (
                                    context_node_pos < 0
                                    or context_node_pos >= len(walk)
                                    or context_node_pos == center_node_pos
                            ):
                                continue

                            context_node_idx = walk[
                                context_node_pos
                            ]

                            curr_pair[1].append(
                                int(context_node_idx)
                            )

                        if len(curr_pair[1]) == 0:
                            continue

                        # src_node = torch.LongTensor(
                        #     [curr_pair[0]]
                        # )
                        #
                        # tgt_nodes = torch.LongTensor(
                        #     curr_pair[1]
                        # )
                        #
                        # optimizer_dw.zero_grad()
                        #
                        # log_pos = sg(
                        #     src_node,
                        #     tgt_nodes,
                        #     neg_sample=False
                        # )
                        #
                        # loss_dw = log_pos
                        #
                        # loss_dw.backward(
                        #     retain_graph=True
                        # )
                        #
                        # optimizer_dw.step()

                # if args.context == 1:
                #     # Construct the pairs for predicting context node
                #     # for each node, treated as center word
                #     # for  center_node_pos in len(nodes_in_G):
                #     center_node_pos= 10
                #
                #     for center_node_pos in range(len(walk)):
                #         if not walk:
                #             print("Walk list is empty.")
                #             continue
                #
                #         if center_node_pos >= len(walk):
                #             print(
                #                 f"center_node_pos {center_node_pos} is out of range for walk list of length {len(walk)}.")
                #             continue
                #
                #     curr_pair = (int(walk[center_node_pos]), [])
                #     for center_node_pos in range(len(walk)):
                #         #curr_pair = (int(walk[center_node_pos]), [])
                #         # for each window position
                #         for w in range(-args.window_size, args.window_size + 1):
                #             context_node_pos = center_node_pos + w
                #             # make soure not jump out sentence
                #             if context_node_pos < 0 or context_node_pos >= len(walk) or center_node_pos == context_node_pos:
                #                 continue
                #             context_node_idx = walk[context_node_pos]
                #             curr_pair[1].append(int(context_node_idx))
                # else:
                #     # first item in the walk is the starting node
                #     curr_pair = (int(walk[0]), [int(context_node_idx) for context_node_idx in walk[1:]])

                if args.ns == 1:
                    neg_nodes = []
                    pos_nodes = set(walk)
                    while len(neg_nodes) < args.walk_length - 1:
                        rand_node = random.randint(0, n_nodes - 1)
                        if rand_node not in pos_nodes:
                            neg_nodes.append(rand_node)
                    neg_nodes = torch.from_numpy(np.array(neg_nodes)).long()

                # Do actual prediction
                src_node = torch.from_numpy(np.array([curr_pair[0]])).long()
                tgt_nodes = torch.from_numpy(np.array(curr_pair[1])).long()
                optimizer_dw.zero_grad()
                # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # weight_decay是L2正则化的强度
                #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.6)


                log_pos = sg(src_node, tgt_nodes, neg_sample=False)
                if args.ns == 1:
                    loss_neg = sg(src_node, neg_nodes, neg_sample=True)
                    loss_dw = log_pos + loss_neg
                else:
                    loss_dw = log_pos
                loss_dw.backward(retain_graph=True)
                cur_dw_loss = loss_dw.item()
                optimizer_dw.step()
        loss = loss_function(preds=model.dc(z), labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        total_loss = loss + (1.0 / (2.0 * torch.exp(log_sigma_dw))) * loss_dw + log_sigma_dw
        total_loss.backward()
        loss.backward()
        cur_loss = loss.item()
        cur_loss = total_loss.item()
        optimizer.step()
#        scheduler.step()

        # hidden_emb = mu.data.numpy()
        hidden_emb = z.detach().cpu().numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)


        if args.dw == 1:
            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, train_loss_dw={:.5f}, val_ap={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss, cur_dw_loss,
                ap_curr, time.time() - t))
        else:
            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, val_ap={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss,
                ap_curr, time.time() - t))

        if (epoch + 1) % 1 == 0:
            tqdm.write("Evaluating intermediate results...")
            kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden_emb)
            predict_labels = kmeans.predict(hidden_emb)
            cm = clustering_metrics(true_labels, predict_labels)
            cm.evaluationClusterModelFromLabel(tqdm)
            ############################################################
            # ADD TORC HERE
            ############################################################
            ############################################################
            # TORC
            ############################################################

            tqdm.write("TORC Clustering Evaluation...")

            # torc_model = TORC()
            # torc_model = TORC(
            #     metric='cosine',
            #     k=15,
            #     cut_std=2.0,
            #     min_cluster_size=10
            # )
            torc_model = TORC(
                metric='cosine',
                k=15,
                cut_ratio=0.03,
                min_cluster_size=15
            )
            # torc_model = TORC(
            #     metric='cosine',
            #     k=10,
            #     cut_ratio=0.15
            # )

            torc_labels = torc_model.fit_predict(
                hidden_emb
            )

            cm_torc = clustering_metrics(
                true_labels,
                torc_labels
            )

            acc_torc, nmi_torc, ari_torc = \
                cm_torc.evaluationClusterModelFromLabel(
                    tqdm
                )

            tqdm.write(
                "TORC ACC={:.5f}, NMI={:.5f}, ARI={:.5f}".format(
                    acc_torc,
                    nmi_torc,
                    ari_torc
                )
            )

            tqdm.write(
                "TORC discovered {} clusters".format(
                    len(np.unique(torc_labels))
                )
            )
            #############################################

            # tqdm.write("TORC Clustering Evaluation...")
            #
            # torc_model = TORC()
            #
            # torc_labels = torc_model.fit_predict(
            #     hidden_emb
            # )
            #
            # cm_torc = clustering_metrics(
            #     true_labels,
            #     torc_labels
            # )
            #
            # cm_torc.evaluationClusterModelFromLabel(
            #     tqdm
            # )
            #
            # roc_score, ap_score = get_roc_score(
            #     hidden_emb,
            #     adj_orig,
            #     test_edges,
            #     test_edges_false
            # )


            acc, nmi, ari = cm.evaluationClusterModelFromLabel(tqdm)
            roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
            tqdm.write('ROC: {}, AP: {}'.format(roc_score, ap_score))
            #outfile = TemporaryFile()
            #np.save(outfile, hidden_emb)
            #np.savetxt('logs/emb_epoch_{}.npy'.format(epoch + 1), hidden_emb,encoding='utf-8')

            if acc > best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_cluster_epoch = epoch + 1
                best_hidden_emb = hidden_emb.copy()


    tqdm.write("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    tqdm.write('Test ROC score: ' + str(roc_score))
    tqdm.write('Test AP score: ' + str(ap_score))
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=0
    ).fit(hidden_emb)

    predict_labels = kmeans.predict(hidden_emb)

    cm = clustering_metrics(
        true_labels,
        predict_labels
    )

    cm.evaluationClusterModelFromLabel(tqdm)
    ############################################################
    # FINAL TORC
    ############################################################

    tqdm.write("Final TORC Clustering Evaluation...")

    # torc_model = TORC()
    # torc_model = TORC(
    #     metric='cosine',
    #     k=15,
    #     cut_std=2.0,
    #     min_cluster_size=10
    # )
    torc_model = TORC(
        metric='cosine',
        k=15,
        cut_ratio=0.03,
        min_cluster_size=15
    )
    # torc_model = TORC(
    #     metric='cosine',
    #     k=10,
    #     cut_ratio=0.25
    # )

    torc_labels = torc_model.fit_predict(
        hidden_emb
    )

    cm_torc = clustering_metrics(
        true_labels,
        torc_labels
    )
    (
        acc_torc,
        nmi_torc,
        ari_torc,
        f1_macro_torc,
        precision_macro_torc,
        recall_macro_torc,
        f1_micro_torc,
        precision_micro_torc,
        recall_micro_torc
    ) = cm_torc.evaluationClusterModelFromLabel1(tqdm)

    # acc_torc, nmi_torc, ari_torc = \
    #     cm_torc.evaluationClusterModelFromLabel(
    #         tqdm
    #     )

    tqdm.write(
        "FINAL TORC ACC={:.5f}, NMI={:.5f}, ARI={:.5f}".format(
            acc_torc,
            nmi_torc,
            ari_torc
        )
    )

    tqdm.write(
        "TORC discovered {} clusters".format(
            len(np.unique(torc_labels))
        )
    )

    # ############################################################
    # # ADD TORC HERE
    # ############################################################
    #
    # tqdm.write("Final TORC Clustering Evaluation...")
    #
    # torc_model = TORC()
    #
    # torc_labels = torc_model.fit_predict(
    #     hidden_emb
    # )
    #
    # cm_torc = clustering_metrics(
    #     true_labels,
    #     torc_labels
    # )
    #
    # cm_torc.evaluationClusterModelFromLabel(
    #     tqdm
    # )
    # kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden_emb)
    # predict_labels = kmeans.predict(hidden_emb)
    # cm = clustering_metrics(true_labels, predict_labels)
    # cm.evaluationClusterModelFromLabel(tqdm)


    if best_hidden_emb is not None:
        kmeans_best = KMeans(n_clusters=args.n_clusters, random_state=0).fit(best_hidden_emb)
        predict_labels_best = kmeans_best.predict(best_hidden_emb)
        cm_best = clustering_metrics(true_labels, predict_labels_best)
        acc_best, nmi_best, ari_best = cm_best.evaluationClusterModelFromLabel(tqdm)

        tqdm.write(
            f"Best Clustering Metrics at epoch {best_cluster_epoch}")



    if args.plot == 1:
        cm.plotClusters(tqdm, hidden_emb, true_labels)


if __name__ == '__main__':
    from torch_geometric.data import Data, DataLoader
    gae_for(args)
