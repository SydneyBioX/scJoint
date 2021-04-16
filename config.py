class Config(object):
	def __init__(self):
	    
		DB = 'db4_control'
        
		if DB =='db4_control':
			# DB info
			self.number_of_class = 7
			self.input_size = 17668
			self.rna_paths = ['data/citeseq_control_rna.npz']
			self.rna_labels = ['data/citeseq_control_cellTypes.txt']		
			self.atac_paths = ['data/asapseq_control_atac.npz']
			self.atac_labels = ['data/asapseq_control_cellTypes.txt'] #Optional. If atac_labels are provided, accuracy after knn would be provided.
			self.rna_protein_paths = ['data/citeseq_control_adt.npz']
			self.atac_protein_paths = ['data/asapseq_control_adt.npz']
			
			# Training config			
			self.batch_size = 256
			self.lr_stage1 = 0.01
			self.lr_stage3 = 0.01
			self.lr_decay_epoch = 20
			self.epochs_stage1 = 20
			self.epochs_stage3 = 20
			self.p = 0.8
			self.embedding_size = 64
			self.momentum = 0.9
			self.center_weight = 1
			self.checkpoint = ''
			self.num_threads = 1
			self.seed = 1
