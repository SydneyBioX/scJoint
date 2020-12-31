from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.linalg import norm
from scipy.special import softmax
import os
import sys


def compute_scores(knn_label, predictions, d, neighbor_indexs, topn):
    num_samples = knn_label.shape[0]

    avg_scores = np.zeros((num_samples, 9))
    for i in range(num_samples):
        cnt = 0
        for j in range(topn):
            if knn_label[i] == np.argmax(predictions[neighbor_indexs[i][j]]):
                avg_scores[i][:] += predictions[neighbor_indexs[i][j]]
                cnt += 1
        avg_scores[i] /= cnt
        assert(knn_label[i] == np.argmax(avg_scores[i]))
    return avg_scores


def compute_scores_weighted(knn_label, predictions, d, neighbor_indexs, topn):
    num_samples = predictions.shape[0]

    avg_scores = np.zeros((num_samples, 9))    
    for i in range(num_samples):
        total_weights = 0.
        for j in range(topn):
            if knn_label[i] == np.argmax(predictions[neighbor_indexs[i][j]]):
                avg_scores[i][:] += (1/(1+d[i][j]))*predictions[neighbor_indexs[i][j]]    
                total_weights += 1/(1+d[i][j]) 
        
        assert(total_weights != 0)
        avg_scores[i] /= total_weights

    return avg_scores


def compute_scores_nearest(knn_label, predictions, d, neighbor_indexs, topn):
    num_samples = predictions.shape[0]

    avg_scores = np.zeros((num_samples, 9))
    for i in range(num_samples):
      d_max = np.max(neighbor_indexs[i]) + 1
      for j in range(topn):
      	if knn_label[i] == np.argmax(predictions[neighbor_indexs[i][j]]) and d[i][j] < d_max:
      		avg_scores[i] = predictions[neighbor_indexs[i][j]]
      		d_max = d[i][j]

    return avg_scores


# read data
print('Read data')

cuurent_dir = os.path.dirname(os.path.abspath(__file__))

rna_data = []
rna_label = []

rna_data_input = np.loadtxt(cuurent_dir + '/'+str(sys.argv[1])+'/rna_embeddings.txt')
rna_data_input = rna_data_input / norm(rna_data_input, axis=1, keepdims=True)
rna_label_input = np.loadtxt(str(sys.argv[2]) + '/train_label.txt')

rna_data_prediction = np.loadtxt(cuurent_dir + '/'+str(sys.argv[1])+'/rna_predictions.txt')
rna_data_prediction_argmax = np.argmax(rna_data_prediction, axis=1)


for i in range(rna_label_input.shape[0]):
    rna_data.append(rna_data_input[i])
    rna_label.append(int(rna_label_input[i]))
rna_label = np.asarray(rna_label)

atac_data_input = np.loadtxt(cuurent_dir + '/'+str(sys.argv[1])+'/atac_embeddings.txt')
atac_data_input = atac_data_input / np.linalg.norm(atac_data_input, axis=1, keepdims=True)


atac_data_prediction = np.loadtxt(cuurent_dir + '/'+str(sys.argv[1])+'/atac_predictions.txt')
atac_pred_label = np.argmax(atac_data_prediction, axis=1)
atac_label_input = np.loadtxt(str(sys.argv[3]) + '/train_label.txt')

total_correct = 0.
total_num = 0.
split = 1
knn = 30
predictions = np.ones(atac_label_input.shape[0]) * -1
for r in range(split):
    atac_data = []
    atac_label = []
    atac_prediction = []
    atac_idx = []

    for i in range(atac_label_input.shape[0]):
        if i % split == r:
            atac_data.append(atac_data_input[i])
            atac_label.append(int(atac_label_input[i]))

            atac_idx.append(i)

    print('KNN start')
    neigh = KNeighborsClassifier(n_neighbors=knn)
    neigh.fit(rna_data, rna_data_prediction_argmax)

    atac_predict = neigh.predict(atac_data)
    d, top10_neighbors = neigh.kneighbors(atac_data, knn)
    avg_score = compute_scores(atac_predict, rna_data_prediction, d, top10_neighbors, knn)
    avg_score = softmax(avg_score, axis=1)
    np.savetxt(cuurent_dir + '/'+str(sys.argv[1])+'/predictions_with_prob.txt', avg_score)
    
    
    for i in range(len(atac_data)):
        predictions[atac_idx[i]] = atac_predict[i]
        assert(atac_predict[i] == np.argmax(avg_score[i]))


    correct = 0.
    batch_num = 0
    new_cnt = 0.
    for i, label in enumerate(atac_label):
        if label < 0:
            continue
        else:
            batch_num += 1

        if label == atac_predict[i]:
            correct += 1

    total_correct += correct
    total_num += batch_num
    print(str(r) + 'nd acc:', correct / batch_num, correct, batch_num)

print('total acc:', total_correct / total_num, total_correct, total_num)

