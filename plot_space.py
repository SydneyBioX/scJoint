import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

from config import Config 


config = Config()
DR_type = 'TSNE' # UMAP or TSNE
subsample_rate = 10


# read data
rna_db0_cnt = 0
rna_cnt = 0
embeddings = []
labels = []

db_name = os.path.basename(config.rna_paths[0]).split('.')[0]
rna_data_input = np.loadtxt('./output/' + db_name + '_embeddings.txt')
rna_label_input = np.loadtxt(config.rna_labels[0])
rna_db0_cnt = rna_label_input.shape[0]//subsample_rate
rna_db0_cnt = 3474

for i in range(1, len(config.rna_paths)):	
	db_name = os.path.basename(config.rna_paths[i]).split('.')[0]
	rna_data_input = np.concatenate((rna_data_input, np.loadtxt('./output/' + db_name + '_embeddings.txt')), 0)
	rna_label_input = np.concatenate((rna_label_input, np.loadtxt(config.rna_labels[i])), 0)
	
for i in range(rna_data_input.shape[0]):
	if i % subsample_rate == 0:
		embeddings.append(rna_data_input[i])
		labels.append(int(rna_label_input[i]))
		rna_cnt += 1


db_name = os.path.basename(config.atac_paths[0]).split('.')[0]
atac_data_input = np.loadtxt('./output/' + db_name + '_embeddings.txt')
atac_label_input = np.loadtxt(config.atac_labels[0])
for i in range(1, len(config.atac_paths)):
	db_name = os.path.basename(config.atac_paths[i]).split('.')[0]
	atac_data_input = np.concatenate((atac_data_input, np.loadtxt('./output/' + db_name + '_embeddings.txt')), 0)
	atac_label_input = np.concatenate((atac_label_input, np.loadtxt(config.atac_labels[i])), 0)

unique_label = np.unique(atac_label_input)

for i in range(atac_data_input.shape[0]):
	if i % subsample_rate == 0:
		embeddings.append(atac_data_input[i])
		labels.append(int(atac_label_input[i]))


embeddings = np.asarray(embeddings)
labels = np.asarray(labels)
		
		
# dimension reduction
print('Total embedding size: ', embeddings.shape)
if DR_type == 'UMAP':
	print('UMAP')
	fit = umap.UMAP(metric='cosine')
	rna_2dim =fit.fit_transform(embeddings)
else:	
	print('TSNE')
	rna_2dim = TSNE().fit_transform(embeddings)



# visualization
x_min, x_max = rna_2dim.min(), rna_2dim.max()
x_norm = (rna_2dim - x_min) / (x_max - x_min)

num_colors = config.number_of_class
cm = plt.get_cmap('gist_rainbow')
plt.figure(figsize=(8,8))
for i in range(x_norm.shape[0]):
	if labels[i] >= 0 and labels[i] in unique_label:
		if i%2 == 0:
			if i < rna_db0_cnt:
				plt.text(x_norm[i,0], x_norm[i,1], 'r0_'+str(labels[i]), color=cm(1/5), fontdict={'weight':'bold', 'size':5})
			elif i < rna_cnt:
				plt.text(x_norm[i,0], x_norm[i,1], 'r1_'+str(labels[i]), color=cm(3/5), fontdict={'weight':'bold', 'size':5})
			else:
				plt.text(x_norm[i,0], x_norm[i,1], 'a_'+str(labels[i]), color=cm(5/5), fontdict={'weight':'bold', 'size':5})
plt.xticks([])
plt.yticks([])
plt.show()


plt.figure(figsize=(8,8))
for i in range(x_norm.shape[0]):
	if labels[i] >= 0 and labels[i] in unique_label:
		if i%2 == 0:
			if i < rna_cnt:
				plt.text(x_norm[i,0], x_norm[i,1], 'r_'+str(labels[i]), color=cm(1.*labels[i]/num_colors), fontdict={'weight':'bold', 'size':5})
			else:
				plt.text(x_norm[i,0], x_norm[i,1], 'a_'+str(labels[i]), color=cm(1.*labels[i]/num_colors), fontdict={'weight':'bold', 'size':5})
plt.xticks([])
plt.yticks([])
plt.show()
