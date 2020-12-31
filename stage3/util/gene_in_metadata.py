import glob
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import os.path
from PIL import ImageFile
import cv2
import random


def load_sparsetxt(file_name):
    feats = np.zeros(17441)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            eles = line.rstrip('\n').split()
            key = int(eles[0])
            val = float(eles[1])
            feats[key] = val
    return feats

def load_sparsetxt2(file_name):
    feats = np.zeros(227)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            eles = line.rstrip('\n').split()
            key = int(eles[0])
            val = float(eles[1])
            feats[key] = val
    return feats


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    print(class_to_idx)
    return classes, class_to_idx

def make_dataset(atac_path, rna_path, atac_protein_path, rna_protein_path):
    data_atac = []
    data_rna = []
    data_atac_protein = []
    data_rna_protein = []
    labels_atac_cell = []
    labels_rna_cell = []

    # ATAC_stim
    with open(os.path.join(atac_protein_path + "/train_data.txt"), "r") as lines:
        for ii, line in enumerate(lines):
            _image = os.path.join(line.rstrip('\n'))
            data_atac_protein.append(_image)
    
    with open(os.path.join(atac_path + "/train_data.txt"), "r") as lines:
        for ii, line in enumerate(lines):
            _image = os.path.join(line.rstrip('\n'))
            data_atac.append(_image)

    with open(os.path.join(atac_path + "/train_label.txt"), "r") as lines:
        for ii, line in enumerate(lines):
            label = int(line.rstrip('\n'))
            labels_atac_cell.append(label)


    # RNA_stim
    with open(os.path.join(rna_protein_path + "/train_data.txt"), "r") as lines:
        for line in lines:
            _data = os.path.join(line.rstrip('\n'))
            data_rna_protein.append(_data)
    
    with open(os.path.join(rna_path + "/train_data.txt"), "r") as lines:
        for line in lines:
            _data = os.path.join(line.rstrip('\n'))
            data_rna.append(_data)

    with open(os.path.join(rna_path + "/train_label.txt"), "r") as lines:
        for line in lines:
            label = int(line.rstrip('\n'))
            labels_rna_cell.append(label)

    cuurent_dir = os.path.dirname(os.path.abspath(__file__))
    labels_pseudo_atac = np.argmax(np.loadtxt(cuurent_dir + '/../../stage1_2/output_txt/0/predictions_with_prob.txt'), axis=1)
    
    return data_atac, data_rna, data_atac_protein, data_rna_protein, labels_atac_cell, labels_rna_cell, labels_pseudo_atac


class MINCDataloder(data.Dataset):
    def __init__(self, atac_path, rna_path, atac_protein_path, rna_protein_path, train=True, transform=None, rna_or_atac='atac'):
        self.transform = transform
        self.train = train
        self.atac_data, self.rna_data, self.atac_protein, self.rna_protein, self.labels_atac, self.labels_rna, self.labels_pseudo_atac = make_dataset(atac_path, rna_path, atac_protein_path, rna_protein_path)
        self.rna_or_atac = rna_or_atac

    def __getitem__(self, index):
        if self.train:
            # get atac data
            rand_idx = random.randint(0, len(self.atac_data) - 1)
            atac_label = self.labels_pseudo_atac[rand_idx]
            atac_image = self.atac_data[rand_idx]
            in_atac_data = ((load_sparsetxt(atac_image)>0).astype(np.float)).reshape((1, 17441))  # binarize data
            
            atac_protein = self.atac_protein[rand_idx]
            in_atac_protein = ((load_sparsetxt2(atac_protein)).astype(np.float)).reshape((1, 227))  # binarize data            
            in_atac_data = np.concatenate((in_atac_data, in_atac_protein), 1)
            

            # get rna data
            rand_idx = random.randint(0, len(self.rna_data) - 1)
            rna_image = self.rna_data[rand_idx]
            rna_label = self.labels_rna[rand_idx]
            in_rna_data = ((load_sparsetxt(rna_image)>0).astype(np.float)).reshape((1, 17441))  # binarize data
            
            rna_protein = self.rna_protein[rand_idx]
            in_rna_protein = ((load_sparsetxt2(rna_protein)).astype(np.float)).reshape((1, 227))  # binarize data            
            in_rna_data = np.concatenate((in_rna_data, in_rna_protein), 1)

            return in_atac_data, in_rna_data, atac_label, rna_label

        else:
            if self.rna_or_atac == 'atac':
                atac_image = self.atac_data[index]
                in_atac_data = ((load_sparsetxt(atac_image)>0).astype(np.float)).reshape((1, 17441))
                
                atac_protein = self.atac_protein[index]
                in_atac_protein = ((load_sparsetxt2(atac_protein)).astype(np.float)).reshape((1, 227))  # binarize data            
                in_atac_data = np.concatenate((in_atac_data, in_atac_protein), 1)
                
                
                atac_cell_label = self.labels_atac[index]
                return in_atac_data, atac_cell_label
            else:
                # get rna data
                rna_data = self.rna_data[index]
                in_rna_data = ((load_sparsetxt(rna_data)>0).astype(np.float)).reshape((1, 17441))
                
                rna_protein = self.rna_protein[index]
                in_rna_protein = ((load_sparsetxt2(rna_protein)).astype(np.float)).reshape((1, 227))  # binarize data            
                in_rna_data = np.concatenate((in_rna_data, in_rna_protein), 1)
                
                
                rna_cell_label = self.labels_rna[index]
                return in_rna_data, rna_cell_label

    def __len__(self):
        if self.train:
            return 20000
        else:
            if self.rna_or_atac == 'atac':
                return len(self.atac_data)
            else:
                return len(self.rna_data)


class Dataloader():
    def __init__(self, args):
        trainset = MINCDataloder(args.atac_path, args.rna_path, args.atac_protein_path, args.rna_protein_path,
                                 train=True)
        testset = MINCDataloder(args.atac_path, args.rna_path, args.atac_protein_path, args.rna_protein_path,
                                train=False, rna_or_atac=args.rnaoratac)

        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
        args.batch_size, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
        args.test_batch_size, shuffle=False, **kwargs)
        self.trainloader = trainloader
        self.testloader = testloader

    def getloader(self):
        return self.trainloader, self.testloader


if __name__ == "__main__":
    data, labels = make_dataset()
