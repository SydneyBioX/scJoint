import numpy as np
import random
import pandas as pd
import os

data_path = "../data/pbmc_control_asapseq_adt_matrix_log.txt"
label_path = "../data/pbmc_control_asapseq_cellTypes.csv"
label_to_idx_path =  "../data/label_to_idx.txt"
output_path = "../data/data_v0/atac_protein_v0/"


    
if not os.path.exists(output_path):
    os.makedirs(output_path)


fp = open(data_path, "r")
lines_brain = fp.readlines()
lines_brain = lines_brain[2:]
fp.close()

print("converting data")
read_vec = np.zeros((4502, 227))
cnt = 0
for line in lines_brain:
    line = line.split(' ')
    row = int(line[0]) - 1
    col = int(line[1]) - 1
    val = float(line[2])

    read_vec[col][row] = val
    cnt += 1

# read label
fp = open(label_path)
lines_brain = fp.readlines()
lines_brain = lines_brain[1:]
fp.close()

brain_labels = []
for line in lines_brain:
    line = line.split(',')
    idx = int(line[0].replace('\"', ''))
    label = line[1].replace('\"', '').replace('\n', '')
    #print(idx, label)
    brain_labels.append(label)

# read label to index
fp = open(label_to_idx_path, "r")
lines_brain = fp.readlines()
label_idx_mapping = {}
for line in lines_brain:
	if line[:-3] == 'unknown ':
		label = 'unknown'
	else:
		label = line[:-3]
	label_idx_mapping[label] = int(line[-2])	

label_idx_mapping['unknown'] = -1
print(label_idx_mapping)

# write data
train_cata = open(output_path + "/train_data.txt", "w")
train_label = open(output_path + "/train_label.txt", "w")

for idx in range(4502):
    train_data = open(output_path + "/train_{}.txt".format(idx), "w")
    for ii, element in enumerate(read_vec[idx, :]):
        if element != 0:
            train_data.write("{} {}\n".format(ii, element))
    train_data.close()
    train_label.write("{}\n".format(label_idx_mapping[brain_labels[idx]]))
    train_cata.write(output_path + "/train_{}.txt\n".format(idx))

train_label.close()
train_cata.close()
