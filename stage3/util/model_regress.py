import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as resnet

class Net_encoder(nn.Module):
    def __init__(self, args):
        super(Net_encoder, self).__init__()
        self.input_size = 17441 + 227
        self.k = 64
        self.f = 64

        self.atac_encoder = nn.Sequential(
            nn.Linear(self.input_size, self.f)
            #nn.Tanh()
        )

        self.BN = nn.Sequential(
            nn.BatchNorm1d(32)
        )

    def forward(self, atac, rna):
        atac = atac.float().view(-1, self.input_size)
        atac_embedding = self.atac_encoder(atac)
        #atac_embedding = atac_embedding / torch.norm(atac_embedding, dim=1, keepdim=True)

        rna = rna.float().view(-1, self.input_size)
        rna_embedding = self.atac_encoder(rna)
        #rna_embedding = rna_embedding / torch.norm(rna_embedding, dim=1, keepdim=True)

        return atac_embedding, rna_embedding


class Net_cell(nn.Module):
    def __init__(self, args):
        super(Net_cell, self).__init__()
        self.rna_cell = nn.Sequential(
            nn.Linear(64, 9)
        )

    def forward(self, atac_embedding, rna_embedding):
        rna_cell_prediction = self.rna_cell(rna_embedding)
        atac_cell_prediction = self.rna_cell(atac_embedding)

        return atac_cell_prediction, rna_cell_prediction
