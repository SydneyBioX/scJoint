#import glob
import torch
import torch.utils.data as data
import numpy as np
import os
import os.path
#import cv2
import random
#import csv
import scipy.sparse

from config import Config


def sparse_mat_reader(file_name):    
    data = scipy.sparse.load_npz(file_name)
    print('Read db:', file_name, ' shape:', data.shape)
    return data, data.shape[1], data.shape[0]


def load_labels(label_file):  # please run parsing_label.py first to get the numerical label file (.txt)
    return np.loadtxt(label_file)



def read_from_file(data_path, label_path = None, protien_path = None):
    data_path = os.path.join(os.path.realpath('.'), data_path)

    data, labels, proteins = None, None, None
    input_size, input_size_protein = 0, 0
    
    data, input_size, sample_num = sparse_mat_reader(data_path)    
    if label_path is not None:        
        label_path = os.path.join(os.path.realpath('.'), label_path)    
        labels = load_labels(label_path)
    if protien_path is not None:
        protien_path = os.path.join(os.path.realpath('.'), protien_path)    
        proteins, input_size_protein, sample_num = sparse_mat_reader(protien_path)
        
    return input_size, sample_num, input_size_protein, data, labels, proteins
    


class Dataloader(data.Dataset):
    def __init__(self, train = True, data_path = None, label_path = None, protien_path = None):
        self.train = train
        self.input_size, self.sample_num, self.input_size_protein, self.data, self.labels, self.proteins = read_from_file(data_path, label_path, protien_path)

    def __getitem__(self, index):
        if self.train:
            # get atac data
            rand_idx = random.randint(0, self.sample_num - 1)
            sample = np.array(self.data[rand_idx].todense())
            in_data = (sample>0).astype(np.float)  # binarize data
            
            if self.proteins is not None:
                sample_protein = np.array(self.proteins[rand_idx].todense())
                in_data = np.concatenate((in_data, sample_protein), 1)
            #in_data = in_data.reshape((1, self.input_size))
            in_label = self.labels[rand_idx]
 
            return in_data, in_label

        else:
            sample = np.array(self.data[index].todense())
            in_data = (sample>0).astype(np.float)  # binarize data

            if self.proteins is not None:
                sample_protein = np.array(self.proteins[index].todense()) 
                in_data = np.concatenate((in_data, sample_protein), 1)
                
            #in_data = in_data.reshape((1, self.input_size))
            in_label = self.labels[index]
 
            return in_data, in_label

    def __len__(self):
        return self.data.shape[0]
                
              
class DataloaderWithoutLabel(data.Dataset):
    def __init__(self, train = True, data_path = None, label_path = None, protien_path = None):
        self.train = train
        self.input_size, self.sample_num, self.input_size_protein, self.data, self.labels, self.proteins = read_from_file(data_path, label_path, protien_path)

    def __getitem__(self, index):
        if self.train:
            # get atac data
            rand_idx = random.randint(0, self.sample_num - 1)
            sample = np.array(self.data[rand_idx].todense())
            in_data = (sample>0).astype(np.float)  # binarize data
            if self.proteins is not None:
                sample_protein = np.array(self.proteins[rand_idx].todense())
                in_data = np.concatenate((in_data, sample_protein), 1)
            #in_data = in_data.reshape((1, self.input_size)) 
            return in_data

        else:
            sample = np.array(self.data[index].todense())
            in_data = (sample>0).astype(np.float)  # binarize data
            if self.proteins is not None:
                sample_protein = np.array(self.proteins[index].todense())
                in_data = np.concatenate((in_data, sample_protein), 1)
                
            #in_data = in_data.reshape((1, self.input_size)) 
            return in_data

    def __len__(self):
        return self.data.shape[0]

                
                


class PrepareDataloader():
    def __init__(self, config):
        self.config = config
        # hardware constraint
        kwargs = {'num_workers': 1, 'pin_memory': True}
        
        # load RNA
        train_rna_loaders = []
        if len(config.rna_paths) == len(config.rna_protein_paths):
            for rna_path, label_path, rna_protein_path in zip(config.rna_paths, config.rna_labels, config.rna_protein_paths):    
                trainset = Dataloader(True, rna_path, label_path, rna_protein_path)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                config.batch_size, shuffle=True, **kwargs)                        
                train_rna_loaders.append(trainloader)
        else:
            for rna_path, label_path in zip(config.rna_paths, config.rna_labels):    
                trainset = Dataloader(True, rna_path, label_path)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                config.batch_size, shuffle=True, **kwargs)                        
                train_rna_loaders.append(trainloader)
                
        
        test_rna_loaders = []
        if len(config.rna_paths) == len(config.rna_protein_paths):
            for rna_path, label_path, rna_protein_path in zip(config.rna_paths, config.rna_labels, config.rna_protein_paths):            
                trainset = Dataloader(False, rna_path, label_path, rna_protein_path)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                config.batch_size, shuffle=False, **kwargs)                        
                test_rna_loaders.append(trainloader)
        else:
            for rna_path, label_path in zip(config.rna_paths, config.rna_labels):            
                trainset = Dataloader(False, rna_path, label_path)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                config.batch_size, shuffle=False, **kwargs)                        
                test_rna_loaders.append(trainloader)
                
        # load ATAC
        train_atac_loaders = []
        self.num_of_atac = 0
        if len(config.atac_paths) == len(config.atac_protein_paths):
            for atac_path, atac_protein_path in zip(config.atac_paths, config.atac_protein_paths):    
                trainset = DataloaderWithoutLabel(True, atac_path, None, atac_protein_path)
                self.num_of_atac += len(trainset)
                
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                config.batch_size, shuffle=True, **kwargs)                        
                train_atac_loaders.append(trainloader)
        else:
            for atac_path in config.atac_paths:    
                trainset = DataloaderWithoutLabel(True, atac_path)
                self.num_of_atac += len(trainset)
                
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                config.batch_size, shuffle=True, **kwargs)                        
                train_atac_loaders.append(trainloader)
                
        test_atac_loaders = []
        if len(config.atac_paths) == len(config.atac_protein_paths):
            for atac_path, atac_protein_path in zip(config.atac_paths, config.atac_protein_paths):    
                trainset = DataloaderWithoutLabel(False, atac_path, None, atac_protein_path)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                config.batch_size, shuffle=False, **kwargs)                        
                test_atac_loaders.append(trainloader)
        else:
            for atac_path in config.atac_paths:    
                trainset = DataloaderWithoutLabel(False, atac_path)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                                config.batch_size, shuffle=False, **kwargs)                        
                test_atac_loaders.append(trainloader)
                                             
  
        self.train_rna_loaders = train_rna_loaders
        self.test_rna_loaders = test_rna_loaders
        self.train_atac_loaders = train_atac_loaders
        self.test_atac_loaders = test_atac_loaders
                    
        
    def getloader(self):
        return self.train_rna_loaders, self.test_rna_loaders, self.train_atac_loaders, self.test_atac_loaders, self.num_of_atac/self.config.batch_size


if __name__ == "__main__":
    config = Config()
    rna_data = Dataloader(True, config.rna_paths[0], config.rna_labels[0])
    print('rna data:', rna_data.input_size, rna_data.input_size_protein, len(rna_data.data))
    
    atac_data = DataloaderWithoutLabel(True, config.atac_paths[0])
    print('atac data:', atac_data.input_size, atac_data.input_size_protein, len(atac_data.data))
    
    
    train_rna_loaders, test_rna_loaders, train_atac_loaders, test_atac_loaders = PrepareDataloader(config).getloader()
    print(len(train_rna_loaders), len(test_atac_loaders))
    
    print(len(train_rna_loaders[1]), len(train_atac_loaders[0]))
