import argparse
import os

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Enrich CNN')
        parser.add_argument('--dataset', type=str, default='minc',
            help='training dataset (default: minc)')
        parser.add_argument('--model', type=str, default='resnet18',
            help='network model type (default: densenet)')
        parser.add_argument('--rnaoratac', type=str, default='atac',
            help='testing data')
        parser.add_argument('--nclass', type=int, default=20, metavar='N',
            help='number of classes (default: 20)')
        # training hyper params
        parser.add_argument('--batch-size', type=int, default=256,
            metavar='N', help='batch size for training (default: 256)')
        parser.add_argument('--test-batch-size', type=int, default=128,
            metavar='N', help='batch size for testing (default: 128)')
        parser.add_argument('--epochs', type=int, default=100, metavar='N',
            help='number of epochs to train (default: 100)')
        parser.add_argument('--start_epoch', type=int, default=1,
            metavar='N', help='the epoch number to start (default: 1)')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
        parser.add_argument('--lr-decay', type=float, default=40, metavar='LD',
            help='epochs for learning rate daecy (default: 40)')
        parser.add_argument('--momentum', type=float, default=0.9,
            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=0.,
            metavar ='M', help='SGD weight decay (default: 1e-4)')
        parser.add_argument('--backbone', type=str, default='resnet50',
            help='backbone name (default: resnet50)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true',
            default=False, help='disables CUDA training')
        parser.add_argument('--plot', action='store_true', default=False,
            help='matplotlib')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
            help='put the path to resuming file if needed')

        parser.add_argument('--train_mode', type=str, default=None,
            help='put the path to resuming file if needed')

        parser.add_argument('--checkname', type=str, default='default',
            help='set the checkpoint name')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
            help='evaluating')
        # lamda
        parser.add_argument('--lamda1', type=float, default=1e-2,
            metavar='N', help='lamda1')
        parser.add_argument('--lamda2', type=float, default=1e-2,
            metavar='N', help='lamda2')
        parser.add_argument('--lamda3', type=float, default=1e-2,
            metavar='N', help='lamda3')
            
        parser.add_argument('--subsample_col', type=int, default=0)
        parser.add_argument('--encoding_p', type=float, default=0.8)
        parser.add_argument('--atac_path', type=str)
        parser.add_argument('--rna_path', type=str)
        parser.add_argument('--atac_protein_path', type=str)
        parser.add_argument('--rna_protein_path', type=str)
            
        self.parser = parser
    def parse(self):
        return self.parser.parse_args()
