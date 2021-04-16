import numpy as np
import scipy.sparse
import h5py
import sys
import os

def h5_reader(file_name):
	h5 = h5py.File(file_name, 'r')
	h5_data = h5['matrix/data']
	print('H5 dataset shape:', h5_data.shape)
	
	return h5_data
	
	
def to_sparse_mat(file_name):
	h5_data = h5_reader(file_name)
	sparse_data = scipy.sparse.csr_matrix(np.array(h5_data).transpose())
	scipy.sparse.save_npz(file_name.replace('h5', 'npz'), sparse_data)
	
	
def data_parsing(rna_h5_files, atac_h5_files):
	for rna_h5_file in rna_h5_files:
		to_sparse_mat(rna_h5_file)
	
	for atac_h5_file in atac_h5_files:
		to_sparse_mat(atac_h5_file)
		
	
def label_parsing(rna_label_files, atac_label_files):
	# 
	total_rna_labels = []
	for label_file in rna_label_files:
		with open(label_file) as fp:
			lines = fp.readlines()
			lines = lines[1:]

			for line in lines:
				line = line.split(',')
				label = line[1].replace('\"', '').replace('\n', '')
				#print(idx, label)
				total_rna_labels.append(label)
	
	# label to idx
	label_idx_mapping = {}
	unique_labels = np.unique(total_rna_labels)
	for i, name in enumerate(unique_labels):		
		label_idx_mapping[name] = i
	print(label_idx_mapping)
	with open(os.path.dirname(rna_label_files[0]) + "/label_to_idx.txt", "w") as fp:
		for key in sorted(label_idx_mapping):
			fp.write(key + " " + str(label_idx_mapping[key]) + '\n')
	
	
	# gen rna label files
	for label_file in rna_label_files:
		with open(label_file) as fp:			
			lines = fp.readlines()
			lines = lines[1:]
			
			with open(label_file.replace('csv','txt'), 'w') as rna_label_f:
				for line in lines:
					line = line.split(',')
					label = line[1].replace('\"', '').replace('\n', '')
					rna_label_f.write(str(label_idx_mapping[label]) + '\n')
					
	# gen atac label files
	for label_file in atac_label_files:
		with open(label_file) as fp:			
			lines = fp.readlines()
			lines = lines[1:]
			
			with open(label_file.replace('csv','txt'), 'w') as rna_label_f:
				for line in lines:
					line = line.split(',')
					label = line[1].replace('\"', '').replace('\n', '')
					if label in label_idx_mapping.keys():
						rna_label_f.write(str(label_idx_mapping[label]) + '\n')
					else:
						rna_label_f.write('-1\n')   # give an unknown label



if __name__ == '__main__':
	rna_h5_files = [] 
	rna_label_files = [] # csv file
	
	atac_h5_files = []
	atac_label_files = []
		
	
	rna_protein_files = [] #h5 file
	atac_protein_files = []
	
	data_parsing(rna_h5_files, atac_h5_files)
	label_parsing(rna_label_files, atac_label_files)
	data_parsing(rna_protein_files, atac_protein_files)

