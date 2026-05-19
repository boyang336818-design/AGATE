import torch
import torch.nn.modules.loss
import torch.nn.functional as F

#
# def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
#     cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)
#
#     # Check if the model is simple Graph Auto-encoder
#     if logvar is None:
#         return cost
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 / n_nodes * torch.mean(torch.sum(
#         1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
#     return cost + KLD

import torch
import torch.nn.functional as F

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight,
                 beta=1.0, alpha=0.5, contrastive=True,
                 labels_for_contrastive=None, margin=1.0):
    # 重构损失
    bce_loss = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)
    cost = bce_loss

    # KL 散度
    if logvar is not None:
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        cost += beta * KLD

    # 对比损失
    if contrastive and labels_for_contrastive is not None:
        # 计算欧氏距离
        dists = torch.cdist(mu, mu, p=2)
        labels = labels_for_contrastive.unsqueeze(1).repeat(1, mu.size(0))
        mask = (labels == labels.T).float().to(mu.device)
        pos_pairs = mask
        neg_pairs = 1 - mask

        contrastive_loss = torch.mean(pos_pairs * dists + neg_pairs * torch.clamp(margin - dists, min=0.0))
        cost += alpha * contrastive_loss

    return cost


