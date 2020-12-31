# scJoint


First, run the following codes in terminal to preprocess the data.

```
cd data_proc
python parse_rna_data.py 
python parse_atac_data.py 
python parse_rna_protein.py 
python parse_atac_protein.py
```

Next, run the following codes to perform stage 1 and 2 of scJoint.

```
cd ../stage1_2
./stage1_2.sh
```

Finally, run the following codes to perform stage 3 of scJoint.

```
cd ../stage3
./stage3.sh
```


