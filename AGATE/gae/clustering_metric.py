from sklearn import metrics
from munkres import Munkres
import numpy as np
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class clustering_metrics:
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        from collections import Counter
        y_true = self.true_label
        y_pred = self.pred_label

        # 如果标签不是从 0 开始的连续整数，先映射
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 获取唯一标签
        true_classes = sorted(np.unique(y_true))
        pred_classes = sorted(np.unique(y_pred))

        num_true = len(true_classes)
        num_pred = len(pred_classes)

        # 构建混淆矩阵（cost matrix）
        cost = np.zeros((num_true, num_pred), dtype=int)
        for i, tc in enumerate(true_classes):
            for j, pc in enumerate(pred_classes):
                cost[i, j] = np.sum((y_true == tc) & (y_pred == pc))

        # 使用 Munkres（匈牙利算法）找最优匹配
        from munkres import Munkres
        m = Munkres()
        # Munkres 要求最小化成本，所以我们用负值（或用最大值减）
        max_cost = cost.max()
        cost_matrix = (max_cost - cost).tolist()  # 转为 profit maximization
        indexes = m.compute(cost_matrix)

        # 创建预测标签到真实标签的映射
        pred_to_true = {}
        for i, j in indexes:
            pred_to_true[pred_classes[j]] = true_classes[i]

        # 对未匹配的预测簇，分配一个新标签（或忽略，这里我们分配为 -1，但会影响 ACC）
        # 更好的做法：只重映射已匹配的，其余保持原样（但 ACC 会偏低）
        new_pred = np.array([pred_to_true.get(p, p) for p in y_pred])

        # 计算指标（注意：未匹配的点可能拉低 ACC）
        acc = metrics.accuracy_score(y_true, new_pred)
        f1_macro = metrics.f1_score(y_true, new_pred, average='macro', zero_division=0)
        precision_macro = metrics.precision_score(y_true, new_pred, average='macro', zero_division=0)
        recall_macro = metrics.recall_score(y_true, new_pred, average='macro', zero_division=0)
        f1_micro = metrics.f1_score(y_true, new_pred, average='micro')
        precision_micro = metrics.precision_score(y_true, new_pred, average='micro')
        recall_micro = metrics.recall_score(y_true, new_pred, average='micro')

        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro
    # def clusteringAcc(self):
    #     # best mapping between true_label and predict label
    #     l1 = list(set(self.true_label))
    #     numclass1 = len(l1)
    #
    #
    #     l2 = list(set(self.pred_label))
    #     numclass2 = len(l2)
    #     #if numclass1 != numclass2:
    #         #print('Class Not equal, Error!!!!')
    #         #return 0
    #
    #     cost = np.zeros((numclass1, numclass2), dtype=int)   #给矩阵赋初始值的时候，经常会用到0矩阵，而Python中，我们使用zero()函数来实现。
    #     for i, c1 in enumerate(l1):
    #         mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
    #         for j, c2 in enumerate(l2):
    #             mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]
    #
    #             cost[i][j] = len(mps_d)
    #
    #     # match two clustering results by Munkres algorithm
    #     m = Munkres()
    #     cost = cost.__neg__().tolist()
    #
    #     indexes = m.compute(cost)
    #
    #     # get the match results
    #     new_predict = np.zeros(len(self.pred_label))
    #     for i, c in enumerate(l1):
    #         # correponding label in l2:
    #         c2 = l2[indexes[i][1]]
    #
    #         # ai is the index with label==c2 in the pred_label list     #ai 是 pred_label 列表中 label==c2 的索引
    #         ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
    #         new_predict[ai] = c
    #
    #     acc = metrics.accuracy_score(self.true_label, new_predict)
    #     f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
    #     precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
    #     recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
    #     f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
    #     precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
    #     recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
    #     return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro
    def evaluationClusterModelFromLabel1(self, tqdm):

        nmi = metrics.normalized_mutual_info_score(
            self.true_label,
            self.pred_label
        )

        ari = metrics.adjusted_rand_score(
            self.true_label,
            self.pred_label
        )

        (
            acc,
            f1_macro,
            precision_macro,
            recall_macro,
            f1_micro,
            precision_micro,
            recall_micro
        ) = self.clusteringAcc()

        tqdm.write(
            'ACC={:.5f}, NMI={:.5f}, ARI={:.5f}, '
            'F1_macro={:.5f}, Precision_macro={:.5f}, Recall_macro={:.5f}, '
            'F1_micro={:.5f}, Precision_micro={:.5f}, Recall_micro={:.5f}'.format(
                acc,
                nmi,
                ari,
                f1_macro,
                precision_macro,
                recall_macro,
                f1_micro,
                precision_micro,
                recall_micro
            )
        )

        return (
            acc,
            nmi,
            ari,
            f1_macro,
            precision_macro,
            recall_macro,
            f1_micro,
            precision_micro,
            recall_micro
        )

    def evaluationClusterModelFromLabel(self, tqdm):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        print(self.clusteringAcc())
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        tqdm.write(
            'ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (
            acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

        # fh = open('recoder.txt', 'a')
        #
        # fh.write(
        #     'ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (
        #     acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))
        # fh.write('\r\n')
        # fh.flush()
        # fh.close()

        return acc, nmi, adjscore

    @staticmethod
    def plot(X, fig, col, size, true_labels):
        ax = fig.add_subplot(1, 1, 1)
        for i, point in enumerate(X):
            ax.scatter(point[0], point[1], s=size, c=col[true_labels[i]])

    def plotClusters(self, tqdm, hidden_emb, true_labels):
        tqdm.write('Start plotting using TSNE...')
        # Doing dimensionality reduction for plotting
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(hidden_emb)
        # Plot figure
        fig = plt.figure()
        self.plot(X_tsne, fig, ['red', 'green', 'blue', 'brown', 'purple', 'yellow', 'pink', 'orange'], 4, true_labels)
        fig.savefig("plot.png")
        tqdm.write("Finished plotting")
