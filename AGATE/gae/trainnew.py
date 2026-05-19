from __future__ import division
from __future__ import print_function
import os, sys
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_auc_score, average_precision_score
import random

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

# For replicating the experiments
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===== 原有模块导入 =====
from gae.model import GCNModelVAE, GCNModelAE, GCNModelAES,GAE, augment_graph
from gae.optimizer import loss_function
# 注意：我们不再直接使用 utils 中的 mask_test_edges，而是使用下面重写的版本
from gae.utils import load_data, preprocess_graph, get_roc_score

try:
    from deepWalk.graph import load_edgelist_from_csr_matrix, build_deepwalk_corpus_iter, build_deepwalk_corpus
    from deepWalk.skipGram import SkipGram

    DEEPWALK_AVAILABLE = True
except ImportError:
    DEEPWALK_AVAILABLE = False
    print("Warning: DeepWalk modules not found.")

from gae.clustering_metric import clustering_metrics

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_ae', help="models used")
parser.add_argument('--dw', type=int, default=1, help="whether to use deepWalk regularization, 0/1")
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='arxiv',
                    help='Dataset: cora, citeseer, photo, computers, arxiv')
parser.add_argument('--walk-length', default=3, type=int, help='Length of the random walk')
parser.add_argument('--window-size', default=3, type=int, help='Window size of skipgram model.')
parser.add_argument('--number-walks', default=5, type=int, help='Number of random walks per node')
parser.add_argument('--full-number-walks', default=0, type=int, help='Number of random walks from each node')
parser.add_argument('--lr_dw', type=float, default=0.01, help='Initial learning rate for regularization.')
parser.add_argument('--context', type=int, default=1, help="whether to use context nodes for skipgram")
parser.add_argument('--ns', type=int, default=0, help="whether to use negative samples for skipgram")
parser.add_argument('--n-clusters', default=10, type=int, help='number of clusters')
parser.add_argument('--plot', type=int, default=0, help="whether to plot the clusters using tsne")
args = parser.parse_args()


# ===== 核心修复：重写 mask_test_edges 以支持稀疏矩阵 (避免 OOM) =====
def mask_test_edges_sparse(adj, test_frac=0.1, val_frac=0.05, prevent_disconnect=True, verbose=True):
    """
    完全基于稀疏矩阵操作的边掩码函数，替代原 utils.py 中的稠密版本。
    专为 ogbn-arxiv 等大图设计。
    """
    if verbose:
        print("Using sparse-safe edge masking (No dense conversion)...")

    # 1. 移除对角线 (自环)
    adj_no_self_loops = adj.copy()
    adj_no_self_loops.setdiag(0)
    adj_no_self_loops.eliminate_zeros()

    # 2. 获取所有存在的边索引 (COO 格式)
    # 只取上三角，因为是无向图，避免重复处理
    adj_triu = sp.triu(adj_no_self_loops, format='coo')
    edge_index = np.vstack((adj_triu.row, adj_triu.col)).T
    num_edges = edge_index.shape[0]

    if verbose:
        print(f"Total edges to split: {num_edges}")

    # 3. 随机打乱边
    rand_perm = np.random.permutation(num_edges)
    edge_index_shuffled = edge_index[rand_perm]

    # 4. 划分测试集、验证集、训练集
    test_size = int(num_edges * test_frac)
    val_size = int(num_edges * val_frac)

    test_edges_pos = edge_index_shuffled[:test_size]
    val_edges_pos = edge_index_shuffled[test_size:test_size + val_size]
    train_edges_pos = edge_index_shuffled[test_size + val_size:]

    # 5. 构建训练用邻接矩阵 (稀疏)
    data = np.ones(train_edges_pos.shape[0])
    adj_train = sp.coo_matrix((data, (train_edges_pos[:, 0], train_edges_pos[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T  # 对称化
    adj_train = adj_train.tocsr()
    adj_train.setdiag(0)  # 确保无自环
    adj_train.eliminate_zeros()

    # 6. 生成负样本 (仅针对验证集和测试集，使用稀疏查找避免稠密化)
    def generate_neg_edges(pos_edges, adj_ref, num_neg=None):
        if num_neg is None:
            num_neg = len(pos_edges)
        neg_edges = []
        n_nodes = adj_ref.shape[0]
        count = 0
        # 将正边转为集合以便快速查找 (u, v) 其中 u < v
        pos_set = set()
        for r, c in pos_edges:
            if r > c: r, c = c, r
            pos_set.add((r, c))
        # 也包含原始图中的边，避免采样到真实边
        coo_ref = adj_ref.tocoo()
        for r, c in zip(coo_ref.row, coo_ref.col):
            if r > c: r, c = c, r
            pos_set.add((r, c))

        while count < num_neg:
            i = np.random.randint(0, n_nodes)
            j = np.random.randint(0, n_nodes)
            if i == j: continue
            if i > j: i, j = j, i
            if (i, j) not in pos_set:
                neg_edges.append([i, j])
                count += 1
        return np.array(neg_edges)

    if verbose:
        print("Generating negative samples for validation and test sets...")

    val_edges_neg = generate_neg_edges(val_edges_pos, adj_no_self_loops)
    test_edges_neg = generate_neg_edges(test_edges_pos, adj_no_self_loops)

    return adj_train, train_edges_pos, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg


# ===== 数据加载函数 =====
def load_new_dataset(name):
    if name in ['photo', 'computers']:
        try:
            from torch_geometric.datasets import Amazon
        except ImportError:
            raise ImportError("Please install torch-geometric")
        dataset = Amazon(root='./data/Amazon', name=name.capitalize())
        data = dataset[0]
        features = data.x.numpy()
        edge_index = data.edge_index.numpy()
        labels = data.y.numpy()
        n_clusters = 8 if name == 'photo' else 10
        N = features.shape[0]
        adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(N, N)).tocsr()
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
        return adj, features, labels, n_clusters, None

    elif name == 'arxiv':
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except ImportError:
            raise ImportError("Please install ogb")
        print("⚠️  WARNING: Running DeepWalk on ogbn-arxiv. Ensure parameters are small.")
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        features = data.x.numpy()
        edge_index = data.edge_index.numpy()
        labels = data.y.numpy().flatten()
        train_idx = split_idx['train'].numpy()
        n_clusters = len(np.unique(labels[train_idx]))
        N = features.shape[0]
        adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(N, N)).tocsr()
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
        return adj, features, labels, n_clusters, train_idx
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ===== 辅助函数 =====
def sampled_loss_function(z, pos_edges, neg_edges, device, pos_weight=1.0):
    z = z.to(device)
    pos_edges = torch.LongTensor(pos_edges).to(device)
    neg_edges = torch.LongTensor(neg_edges).to(device)
    pos_scores = torch.sum(z[pos_edges[:, 0]] * z[pos_edges[:, 1]], dim=1)
    neg_scores = torch.sum(z[neg_edges[:, 0]] * z[neg_edges[:, 1]], dim=1)
    pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean() * pos_weight
    neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-8).mean()
    return pos_loss + neg_loss


def generate_neg_edges_batch(adj_orig, num_neg, n_nodes):
    neg_edges = []
    count = 0
    while count < num_neg:
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        if i == j: continue
        if adj_orig[i, j] == 0:
            neg_edges.append([i, j])
            count += 1
    return neg_edges


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))

    # 数据加载
    if args.dataset_str in ['cora', 'citeseer']:
        adj, features, y_test, tx, ty, test_maks, true_labels = load_data(args.dataset_str)
        train_idx = None
        is_large_graph = False
        if args.dataset_str == 'cora':
            args.n_clusters = 7
        elif args.dataset_str == 'citeseer':
            args.n_clusters = 6
    elif args.dataset_str in ['photo', 'computers', 'arxiv']:
        adj, features, true_labels, n_clusters, train_idx = load_new_dataset(args.dataset_str)
        args.n_clusters = n_clusters
        y_test, tx, ty, test_maks = None, None, None, None
        is_large_graph = (args.dataset_str == 'arxiv')
    else:
        raise ValueError("Unsupported dataset")

    n_nodes, feat_dim = features.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    adj_orig = adj.copy()
    adj_orig.setdiag(0)
    adj_orig.eliminate_zeros()

    # 【关键修复】：使用重写的稀疏版本 mask_test_edges_sparse
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_sparse(adj)
    adj = adj_train

    # DeepWalk 准备
    G = None
    if args.dw == 1:
        if not DEEPWALK_AVAILABLE:
            print("🛑 DeepWalk module not found.")
            sys.exit(1)
        else:
            print('Using deepWalk regularization...')
            G = load_edgelist_from_csr_matrix(adj_orig, undirected=True)
            print(f"Nodes: {len(G.nodes())}, Walks/node: {args.number_walks}, Length: {args.walk_length}")

    # Preprocessing
    adj_norm = preprocess_graph(adj)

    # 大图禁止 toarray()
    if is_large_graph:
        print("⚠️ Large graph mode: Skipping dense adj_label.")
        adj_label = None
        pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
        norm = 1.0
    else:
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray()).to(device)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Model
    if args.model == 'gcn_ae':
        model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        # model = GAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    else:
        model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        # model = GAE(feat_dim, args.hidden1, args.hidden2, args.dropout)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(device)
    features_tensor = torch.FloatTensor(features).to(device)
    adj_norm = adj_norm.to(device)

    # DeepWalk model
    sg = None
    optimizer_dw = None
    nodes_in_G = chunks = None

    if args.dw == 1:
        sg = SkipGram(args.hidden2, adj.shape[0]).to(device)
        optimizer_dw = optim.Adam(sg.parameters(), lr=args.lr_dw)
        nodes_in_G = list(G.nodes())
        chunks = max(len(nodes_in_G) // max(1, args.number_walks), 100)
        random.Random(SEED).shuffle(nodes_in_G)

    best_acc = best_nmi = best_ari = 0.0
    best_hidden_emb = None
    batch_size = 2048 if is_large_graph else 0

    for epoch in tqdm(range(args.epochs)):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        if args.dw == 1: sg.train()

        z, mu, logvar = model(features_tensor, adj_norm)

        cur_dw_loss = 0.0
        if args.dw == 1:
            if args.full_number_walks > 0:
                walks = build_deepwalk_corpus(G, num_paths=args.full_number_walks, path_length=args.walk_length,
                                              alpha=0, rand=random.Random(SEED))
            else:
                walks = build_deepwalk_corpus_iter(G, num_paths=args.number_walks, path_length=args.walk_length,
                                                   alpha=0, rand=random.Random(SEED), chunk=epoch % chunks,
                                                   nodes=nodes_in_G)

            for walk in walks:
                if args.context == 1:
                    curr_pair_list = []
                    for center_node_pos in range(len(walk)):
                        ctx_list = []
                        for w in range(-args.window_size, args.window_size + 1):
                            context_node_pos = center_node_pos + w
                            if 0 <= context_node_pos < len(walk) and center_node_pos != context_node_pos:
                                ctx_list.append(int(walk[context_node_pos]))
                        if ctx_list:
                            curr_pair_list.append((int(walk[center_node_pos]), ctx_list))
                else:
                    curr_pair_list = [(int(walk[0]), [int(x) for x in walk[1:]])]

                for src, tgts in curr_pair_list:
                    if args.ns == 1:
                        neg_nodes = []
                        pos_nodes = set(walk)
                        while len(neg_nodes) < len(tgts):
                            rand_node = random.randint(0, n_nodes - 1)
                            if rand_node not in pos_nodes: neg_nodes.append(rand_node)
                        neg_nodes = torch.LongTensor(neg_nodes).to(device)

                    src_node = torch.LongTensor([src]).to(device)
                    tgt_nodes = torch.LongTensor(tgts).to(device)

                    optimizer_dw.zero_grad()
                    log_pos = sg(src_node, tgt_nodes, neg_sample=False)
                    if args.ns == 1:
                        loss_neg = sg(src_node, neg_nodes, neg_sample=True)
                        loss_dw = log_pos + loss_neg
                    else:
                        loss_dw = log_pos

                    loss_dw.backward(retain_graph=True)
                    cur_dw_loss += loss_dw.item()
                    optimizer_dw.step()

        # Loss 计算
        if is_large_graph:
            if len(train_edges) > batch_size:
                pos_idx = np.random.choice(len(train_edges), batch_size, replace=False)
                pos_batch = train_edges[pos_idx]
            else:
                pos_batch = train_edges
            neg_batch = generate_neg_edges_batch(adj_orig, len(pos_batch), n_nodes)
            loss = sampled_loss_function(mu, pos_batch, neg_batch, device, pos_weight=pos_weight)
        else:
            loss = loss_function(preds=model.dc(z), labels=adj_label, mu=mu, logvar=logvar, n_nodes=n_nodes, norm=norm,
                                 pos_weight=pos_weight)

        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.detach().cpu().numpy()

        if is_large_graph:
            roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, None, is_large_graph=True,
                                              num_neg_samples=1000)
        else:
            roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        if args.dw == 1:
            tqdm.write(
                f"Epoch: {epoch + 1}, GAE={cur_loss:.5f}, DW={cur_dw_loss:.5f}, AP={ap_curr:.5f}, T={time.time() - t:.2f}s")
        else:
            tqdm.write(f"Epoch: {epoch + 1}, GAE={cur_loss:.5f}, AP={ap_curr:.5f}, T={time.time() - t:.2f}s")

        # Clustering
        if (epoch + 1) % 1 == 0:
            if is_large_graph and train_idx is not None:
                emb_sub = hidden_emb[train_idx]
                kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(emb_sub)
                predict_labels = kmeans.predict(emb_sub)
                cm = clustering_metrics(true_labels[train_idx], predict_labels)
            else:
                kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden_emb)
                predict_labels = kmeans.predict(hidden_emb)
                cm = clustering_metrics(true_labels, predict_labels)

            acc, nmi, ari = cm.evaluationClusterModelFromLabel(tqdm)
            if acc > best_acc:
                best_acc, best_nmi, best_ari = acc, nmi, ari
                best_hidden_emb = hidden_emb.copy()

    # Final Test
    if is_large_graph:
        roc_score, ap_score = get_roc_score(best_hidden_emb, adj_orig, test_edges, None, is_large_graph=True,
                                            num_neg_samples=2000)
    else:
        roc_score, ap_score = get_roc_score(best_hidden_emb, adj_orig, test_edges, test_edges_false)

    print(f'Test ROC: {roc_score:.4f}, AP: {ap_score:.4f}')

    if is_large_graph and train_idx is not None:
        emb_sub = best_hidden_emb[train_idx]
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(emb_sub)
        predict_labels = kmeans.predict(emb_sub)
        cm = clustering_metrics(true_labels[train_idx], predict_labels)
    else:
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(best_hidden_emb)
        predict_labels = kmeans.predict(best_hidden_emb)
        cm = clustering_metrics(true_labels, predict_labels)
    cm.evaluationClusterModelFromLabel(tqdm)


# 本地覆盖 get_roc_score
def get_roc_score(emb, adj_orig, edges_pos, edges_neg, is_large_graph=False, num_neg_samples=1000):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    preds = [sigmoid(emb[e[0]] @ emb[e[1]]) for e in edges_pos]
    preds_neg = []
    if edges_neg is not None:
        preds_neg = [sigmoid(emb[e[0]] @ emb[e[1]]) for e in edges_neg]
    else:
        num_pos = len(edges_pos)
        neg_count = 0
        while neg_count < min(num_neg_samples, num_pos):
            i, j = np.random.randint(0, emb.shape[0], 2)
            if i == j or adj_orig[i, j] != 0: continue
            preds_neg.append(sigmoid(emb[i] @ emb[j]))
            neg_count += 1
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    return roc_auc_score(labels_all, preds_all), average_precision_score(labels_all, preds_all)


if __name__ == '__main__':
    gae_for(args)















# from __future__ import division
# from __future__ import print_function
# import os, sys
# import numpy as np
# import scipy.sparse as sp
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from tqdm import tqdm
# import time
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import pairwise_distances
# from sklearn.metrics import roc_auc_score, average_precision_score
# import random
#
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
#
# # For replicating the experiments
# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)
#
# # ===== 原有模块（完全保留）=====
# from gae.model import GCNModelVAE, GCNModelAE, GCNModelAES, augment_graph
# from gae.optimizer import loss_function
# from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
# from deepWalk.graph import load_edgelist_from_csr_matrix, build_deepwalk_corpus_iter, build_deepwalk_corpus
# from deepWalk.skipGram import SkipGram
# from gae.clustering_metric import clustering_metrics
#
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='gcn_ae', help="models used")
# parser.add_argument('--dw', type=int, default=1, help="whether to use deepWalk regularization, 0/1")
# parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
# parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
# parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
# parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
# parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--dataset-str', type=str, default='citeseer',
#                     help='Dataset: cora, citeseer, photo, computers, arxiv')
# parser.add_argument('--walk-length', default=30, type=int, help='Length of the random walk started at each node')
# parser.add_argument('--window-size', default=30, type=int, help='Window size of skipgram model.')
# parser.add_argument('--number-walks', default=50, type=int, help='Number of random walks to start at each node')
# parser.add_argument('--full-number-walks', default=0, type=int, help='Number of random walks from each node')
# parser.add_argument('--lr_dw', type=float, default=0.01, help='Initial learning rate for regularization.')
# parser.add_argument('--context', type=int, default=1, help="whether to use context nodes for skipgram")
# parser.add_argument('--ns', type=int, default=0, help="whether to use negative samples for skipgram")
# parser.add_argument('--n-clusters', default=40, type=int, help='number of clusters')
# parser.add_argument('--plot', type=int, default=0, help="whether to plot the clusters using tsne")
# args = parser.parse_args()
#
#
# # ===== 新增：加载 Photo / Computers / ogbn-arxiv =====
# def load_new_dataset(name):
#     if name in ['photo', 'computers']:
#         try:
#             from torch_geometric.datasets import Amazon
#         except ImportError:
#             raise ImportError("Please install torch-geometric to use 'photo' or 'computers'")
#         dataset = Amazon(root='./data/Amazon', name=name.capitalize())
#         data = dataset[0]
#         features = data.x.numpy()
#         edge_index = data.edge_index.numpy()
#         labels = data.y.numpy()
#         n_clusters = 8 if name == 'photo' else 10
#
#         N = features.shape[0]
#         adj = sp.coo_matrix(
#             (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
#             shape=(N, N)
#         ).tocsr()
#         adj = adj + adj.T
#         adj.data = np.ones_like(adj.data)  # remove duplicates
#         return adj, features, labels, n_clusters, None
#
#     elif name == 'arxiv':
#         try:
#             from ogb.nodeproppred import PygNodePropPredDataset
#         except ImportError:
#             raise ImportError("Please install ogb to use 'arxiv'")
#         print("⚠️  WARNING: ogbn-arxiv is too large for deepWalk. Disabling dw=1 automatically.")
#         dataset = PygNodePropPredDataset(name='ogbn-arxiv')
#         data = dataset[0]
#         split_idx = dataset.get_idx_split()
#         features = data.x.numpy()
#         edge_index = data.edge_index.numpy()
#         labels = data.y.numpy().flatten()
#         train_idx = split_idx['train'].numpy()
#         n_clusters = len(np.unique(labels[train_idx]))
#
#         N = features.shape[0]
#         adj = sp.coo_matrix(
#             (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
#             shape=(N, N)
#         ).tocsr()
#         adj = adj + adj.T
#         adj.data = np.ones_like(adj.data)
#         return adj, features, labels, n_clusters, train_idx
#
#     else:
#         raise ValueError(f"Unknown dataset: {name}")
#
#
# def gae_for(args):
#     print("Using {} dataset".format(args.dataset_str))
#
#     # ===== 数据加载分支 =====
#     if args.dataset_str in ['cora', 'citeseer']:
#         adj, features, y_test, tx, ty, test_maks, true_labels = load_data(args.dataset_str)
#         train_idx = None
#         if args.dataset_str == 'cora':
#             args.n_clusters = 7
#         elif args.dataset_str == 'citeseer':
#             args.n_clusters = 6
#     elif args.dataset_str in ['photo', 'computers', 'arxiv']:
#         adj, features, true_labels, n_clusters, train_idx = load_new_dataset(args.dataset_str)
#         args.n_clusters = n_clusters
#         # Dummy placeholders to match original interface
#         y_test, tx, ty, test_maks = None, None, None, None
#     else:
#         raise ValueError("Unsupported dataset")
#
#     n_nodes, feat_dim = features.shape
#     adj_orig = adj.copy()
#     adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
#     adj_orig.eliminate_zeros()
#
#     adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
#     adj = adj_train
#
#     # ===== DeepWalk 准备（仅在非 arxiv 且 dw=1 时启用）=====
#     G = None
#     if args.dw == 1:
#         if args.dataset_str == 'arxiv':
#             print("🛑 DeepWalk disabled for ogbn-arxiv due to size. Setting dw=0.")
#             args.dw = 0
#         else:
#             print('Using deepWalk regularization...')
#             G = load_edgelist_from_csr_matrix(adj_orig, undirected=True)
#             print("Number of nodes: {}".format(len(G.nodes())))
#             num_walks = len(G.nodes()) * args.number_walks
#             print("Number of walks: {}".format(num_walks))
#             data_size = num_walks * args.walk_length
#             print("Data size (walks*length): {}".format(data_size))
#
#     # Preprocessing
#     adj_norm = preprocess_graph(adj)
#     adj_label = adj_train + sp.eye(adj_train.shape[0])
#     adj_label = torch.FloatTensor(adj_label.toarray())
#
#     pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#     norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
#
#     # Model
#     if args.model == 'gcn_ae':
#         model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
#     else:
#         model = GCNModelAES(feat_dim, args.hidden1, args.hidden2, args.dropout)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#
#     # DeepWalk model
#     sg = None
#     optimizer_dw = None
#     nodes_in_G = chunks = None
#     if args.dw == 1:
#         sg = SkipGram(args.hidden2, adj.shape[0])
#         optimizer_dw = optim.Adam(sg.parameters(), lr=args.lr_dw)
#         nodes_in_G = list(G.nodes())
#         chunks = len(nodes_in_G) // args.number_walks
#         random.Random(SEED).shuffle(nodes_in_G)
#
#     # Training loop
#     best_acc = best_nmi = best_ari = 0.0
#     best_hidden_emb = None
#
#     for epoch in tqdm(range(args.epochs)):
#         t = time.time()
#         model.train()
#         optimizer.zero_grad()
#         if args.dw == 1:
#             sg.train()
#
#         z, mu, logvar = model(torch.FloatTensor(features), adj_norm)
#
#         cur_dw_loss = 0.0
#         if args.dw == 1:
#             if args.full_number_walks > 0:
#                 walks = build_deepwalk_corpus(G, num_paths=args.full_number_walks,
#                                               path_length=args.walk_length, alpha=0,
#                                               rand=random.Random(SEED))
#             else:
#                 walks = build_deepwalk_corpus_iter(G, num_paths=args.number_walks,
#                                                    path_length=args.walk_length, alpha=0,
#                                                    rand=random.Random(SEED),
#                                                    chunk=epoch % chunks,
#                                                    nodes=nodes_in_G)
#             for walk in walks:
#                 if args.context == 1:
#                     center_node_pos = 0  # fix bug: move inside loop
#                     curr_pair = (int(walk[center_node_pos]), [])
#                     for center_node_pos in range(len(walk)):
#                         curr_pair = (int(walk[center_node_pos]), [])
#                         for w in range(-args.window_size, args.window_size + 1):
#                             context_node_pos = center_node_pos + w
#                             if context_node_pos < 0 or context_node_pos >= len(
#                                     walk) or center_node_pos == context_node_pos:
#                                 continue
#                             context_node_idx = walk[context_node_pos]
#                             curr_pair[1].append(int(context_node_idx))
#                 else:
#                     curr_pair = (int(walk[0]), [int(x) for x in walk[1:]])
#
#                 if args.ns == 1:
#                     neg_nodes = []
#                     pos_nodes = set(walk)
#                     while len(neg_nodes) < args.walk_length - 1:
#                         rand_node = random.randint(0, n_nodes - 1)
#                         if rand_node not in pos_nodes:
#                             neg_nodes.append(rand_node)
#                     neg_nodes = torch.LongTensor(neg_nodes)
#
#                 src_node = torch.LongTensor([curr_pair[0]])
#                 tgt_nodes = torch.LongTensor(curr_pair[1])
#                 optimizer_dw.zero_grad()
#                 log_pos = sg(src_node, tgt_nodes, neg_sample=False)
#                 if args.ns == 1:
#                     loss_neg = sg(src_node, neg_nodes, neg_sample=True)
#                     loss_dw = log_pos + loss_neg
#                 else:
#                     loss_dw = log_pos
#                 loss_dw.backward(retain_graph=True)
#                 cur_dw_loss = loss_dw.item()
#                 optimizer_dw.step()
#
#         loss = loss_function(preds=model.dc(z), labels=adj_label,
#                              mu=mu, logvar=logvar, n_nodes=n_nodes,
#                              norm=norm, pos_weight=pos_weight)
#         loss.backward()
#         cur_loss = loss.item()
#         optimizer.step()
#
#         hidden_emb = mu.detach().cpu().numpy()
#         roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
#
#         if args.dw == 1:
#             tqdm.write("Epoch: {}, train_loss_gae={:.5f}, train_loss_dw={:.5f}, val_ap={:.5f}, time={:.5f}".format(
#                 epoch + 1, cur_loss, cur_dw_loss, ap_curr, time.time() - t))
#         else:
#             tqdm.write("Epoch: {}, train_loss_gae={:.5f}, val_ap={:.5f}, time={:.5f}".format(
#                 epoch + 1, cur_loss, ap_curr, time.time() - t))
#
#         # Clustering evaluation
#         if (epoch + 1) % 1 == 0:
#             if args.dataset_str == 'arxiv' and train_idx is not None:
#                 # Only evaluate on training nodes for arxiv
#                 emb_sub = hidden_emb[train_idx]
#                 kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(emb_sub)
#                 predict_labels = kmeans.predict(emb_sub)
#                 cm = clustering_metrics(true_labels[train_idx], predict_labels)
#             else:
#                 kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden_emb)
#                 predict_labels = kmeans.predict(hidden_emb)
#                 cm = clustering_metrics(true_labels, predict_labels)
#
#             acc, nmi, ari = cm.evaluationClusterModelFromLabel(tqdm)
#             if acc > best_acc:
#                 best_acc, best_nmi, best_ari = acc, nmi, ari
#                 best_hidden_emb = hidden_emb.copy()
#
#     # Final test
#     roc_score, ap_score = get_roc_score(best_hidden_emb, adj_orig, test_edges, test_edges_false)
#     tqdm.write('Test ROC score: ' + str(roc_score))
#     tqdm.write('Test AP score: ' + str(ap_score))
#
#     if args.dataset_str == 'arxiv' and train_idx is not None:
#         emb_sub = best_hidden_emb[train_idx]
#         kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(emb_sub)
#         predict_labels = kmeans.predict(emb_sub)
#         cm = clustering_metrics(true_labels[train_idx], predict_labels)
#     else:
#         kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(best_hidden_emb)
#         predict_labels = kmeans.predict(best_hidden_emb)
#         cm = clustering_metrics(true_labels, predict_labels)
#     cm.evaluationClusterModelFromLabel(tqdm)
#
#
# if __name__ == '__main__':
#     gae_for(args)