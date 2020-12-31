RNA_PATH="../data/data_v0/rna_v0"
ATAC_PATH="../data/data_v0/atac_v0"
RNA_PROTEIN_PATH="../data/data_v0/rna_protein_v0"
ATAC_PROTEIN_PATH="../data/data_v0/atac_protein_v0"


mkdir output_txt/0

#stage 1
CUDA_VISIBLE_DEVICES=0 python3 train_atac.py --model model_regress --subsample_col 0 --encoding_p 0.8 --rnaoratac 'atac' --batch-size 256\
 --train_mode all --test-batch-size 256 --lr 0.01 --epochs 10 --nclass 9 --checkname test\
 --rna_path $RNA_PATH --atac_path $ATAC_PATH --rna_protein_path $RNA_PROTEIN_PATH --atac_protein_path $ATAC_PROTEIN_PATH
 
CUDA_VISIBLE_DEVICES=0 python3 train_atac.py --model model_regress --subsample_col 0 --rnaoratac 'atac' --test-batch-size 256\
 --train_mode test_print --resume ./runs/minc/model_regress/test/checkpoint.pth.tar\
 --rna_path $RNA_PATH --atac_path $ATAC_PATH --rna_protein_path $RNA_PROTEIN_PATH --atac_protein_path $ATAC_PROTEIN_PATH
 
CUDA_VISIBLE_DEVICES=0 python3 train_atac.py --model model_regress --subsample_col 0 --rnaoratac 'rna' --test-batch-size 256\
 --train_mode test_print --resume ./runs/minc/model_regress/test/checkpoint.pth.tar\
 --rna_path $RNA_PATH --atac_path $ATAC_PATH --rna_protein_path $RNA_PROTEIN_PATH --atac_protein_path $ATAC_PROTEIN_PATH

#stage 2
CUDA_VISIBLE_DEVICES=0 python3 output_txt/knn_prob.py 0 $RNA_PATH $ATAC_PATH


