import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def _one_hot(tensor, num):
    b = list(tensor.size())[0]
    onehot = torch.cuda.FloatTensor(b, num).fill_(0)
    ones = torch.cuda.FloatTensor(b, num).fill_(1)
    out = onehot.scatter_(1, torch.unsqueeze(tensor, 1), ones)
    return out


class L1regularization(nn.Module):
    def __init__(self, weight_decay=1.):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay

        return regularization_loss


def cov(m):
    m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def reduction_loss(embedding, identity_matrix_cov, size):
    loss = F.l1_loss(cov(embedding), identity_matrix_cov)
    loss = loss + 1 / torch.mean(
        torch.abs(embedding - torch.mean(embedding, dim=0).view(1, size).repeat(embedding.size()[0], 1)))
    loss = loss + torch.mean(torch.abs(embedding))
    return loss


def cosine_sim(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = torch.matmul(x, torch.transpose(y, 0, 1))

    return sim



class EncodingLoss(nn.Module):
    def __init__(self, dim=64, p =0.8):
        super(EncodingLoss, self).__init__()
        self.identity_matrix_cov = torch.tensor(np.identity(dim)).float().cuda()
        self.p = p 
		
    def forward(self, atac_embedding, rna_embedding):
        loss_atac = reduction_loss(atac_embedding, self.identity_matrix_cov, 64)
        loss_rna = reduction_loss(rna_embedding, self.identity_matrix_cov, 64)

        top_k_sim = torch.topk(
            torch.max(cosine_sim(atac_embedding, rna_embedding), dim=1)[0],
            int(atac_embedding.shape[0] * self.p))
        dist_atac = torch.mean(top_k_sim[0])


        loss = loss_atac + loss_rna - dist_atac
        return loss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=20, feat_dim=64, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        distmat = torch.sqrt(distmat)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = torch.mean(dist.clamp(min=1e-12, max=1e+12))

        return loss


class CellLoss(nn.Module):
    def __init__(self):
        super(CellLoss, self).__init__()

    def forward(self, rna_cell_out, rna_cell_label):
        rna_cell_loss = F.cross_entropy(rna_cell_out, rna_cell_label)
        return rna_cell_loss


class SampleLoss(nn.Module):
    def __init__(self, lamda1, lamda2):
        super(SampleLoss, self).__init__()
        self.l1 = torch.tensor(lamda1).cuda()
        self.l2 = torch.tensor(lamda2).cuda()

    def forward(self, atac_predictions, atac_pos_predictions, atac_neg_predictions):
        sample_loss = torch.mean(F.cosine_similarity(atac_predictions, atac_neg_predictions)) - torch.mean(
            F.cosine_similarity(atac_predictions, atac_pos_predictions))
        return sample_loss


class PairLoss(nn.Module):
    def __init__(self):
        super(PairLoss, self).__init__()

    def forward(self, embedding1, embedding2):
        pair_loss = F.mse_loss(embedding1, embedding2)
        # pair_loss = 1 - torch.mean(F.cosine_similarity(embedding1, embedding2))
        return pair_loss
