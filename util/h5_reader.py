import scipy.sparse as sp_sparse
import time 
from sys import getsizeof
import numpy as np
import ctypes
import h5py


class H5ls:
    def __init__(self):
        self.arrays_metadata = {}
        
    def __call__(self, name, item):     
        is_dataset = isinstance(item, h5py.Dataset)
        if is_dataset:
            offset = item.id.get_offset()
            if offset is not None:
                print(name,offset,item.shape,item.dtype)
                self.arrays_metadata[name] = dict(offset=offset, shape=item.shape, dtype=item.dtype)
            else:
                print('could not get offset, probably not a continuous array')
                

def get_h5_file_dataset_offset(h5_binary_path, h5ls):
    h5file = h5py.File(h5_binary_path, 'r')
    h5file.visititems(h5ls)
    h5file.close()

    metadata_offset = h5ls.arrays_metadata['rna/metadata']['offset']
    data_offset = h5ls.arrays_metadata['rna/data']['offset']
    indptr_offset = h5ls.arrays_metadata['rna/indptr']['offset']
    indices_offset = h5ls.arrays_metadata['rna/indices']['offset']

    return metadata_offset, data_offset, indptr_offset, indices_offset
    
    
def read_sparse_matrix_shape_C(h5_binary_path,metadata_offset):
    c_lib = ctypes.cdll.LoadLibrary('./util/libutility.so')
    c_lib.get_sparse_matrix_shape.argtypes = [ctypes.c_int,
                                              ctypes.POINTER(ctypes.c_int8),
                                              ctypes.POINTER(ctypes.c_int),
                                              ctypes.POINTER(ctypes.c_int)]
    c_lib.get_sparse_matrix_shape.restype = None

    row_num = np.zeros(1, dtype=np.int32).flatten()
    col_num = np.zeros(1, dtype=np.int32).flatten()

    h5_binary_path_array = np.asarray([ord(i) for i in h5_binary_path]+[0],dtype=np.int8)
    c_lib.get_sparse_matrix_shape(ctypes.c_int(metadata_offset),
                                         np.ctypeslib.as_ctypes(h5_binary_path_array),
                                         np.ctypeslib.as_ctypes(row_num),
                                         np.ctypeslib.as_ctypes(col_num))
                                         
    return row_num[0],col_num[0]                                     
   

def read_sparse_matrix_data_C(index_list,width,h5_binary_path,data_offset,indptr_offset,indices_offset,metadata_offset):

    c_lib = ctypes.cdll.LoadLibrary('./util/libutility.so')

    c_lib.read_sparse_matrix_by_index_v2.argtypes = [ctypes.POINTER(ctypes.c_int),
                                                     ctypes.c_int,
                                                     ctypes.POINTER(ctypes.c_float),
                                                     ctypes.POINTER(ctypes.c_int8),
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_int,
                                                     ctypes.c_int]
    c_lib.read_sparse_matrix_by_index.restype = None

    row_num = len(index_list)

    index_list_flatten = np.asarray(index_list, dtype=np.int32).flatten()
    out = np.zeros(row_num*width, dtype=np.float32).flatten()
    h5_binary_path_array = np.asarray([ord(i) for i in h5_binary_path]+[0],dtype=np.int8)
    c_lib.read_sparse_matrix_by_index_v2(np.ctypeslib.as_ctypes(index_list_flatten),
                                         ctypes.c_int(row_num),
                                         np.ctypeslib.as_ctypes(out),
                                         np.ctypeslib.as_ctypes(h5_binary_path_array),
                                         ctypes.c_int(data_offset),
                                         ctypes.c_int(indptr_offset),
                                         ctypes.c_int(indices_offset),
                                         ctypes.c_int(metadata_offset))


    out = out.reshape(row_num,width)

    return out

                                         
class H5_Reader:
    def __init__(self, file_path):
        h5ls = H5ls()
        self.h5_binary_path = file_path
        self.metadata_offset, self.data_offset, self.indptr_offset, self.indices_offset = get_h5_file_dataset_offset(file_path, h5ls)
        self.row_num, self.col_num = read_sparse_matrix_shape_C(self.h5_binary_path, self.metadata_offset)
        
    def get_row(self, index):
        c_row = read_sparse_matrix_data_C([index], self.col_num, self.h5_binary_path, self.data_offset, self.indptr_offset, self.indices_offset, self.metadata_offset)
        return c_row[0]

    


if __name__ == "__main__":
    row_list_size = 64

    h5_binary_path = 'mytestfile.hdf5'
    h5_reader = H5_Reader(h5_binary_path)


    random_list = np.random.randint(h5_reader.row_num, size=row_list_size)

    time1 = time.time()
    for i in range(row_list_size):
        c_row = h5_reader.get_row(random_list[i])
    time2 = time.time()
    total_time = time2-time1
    print('C load row time',round(total_time*1000),'ms')

    with h5py.File("mytestfile.hdf5", 'r') as hf:
        indices = hf[u'rna/indices'][:] # <np.array>
        indptr = hf[u'rna/indptr'][:] # <np.array>
        data = hf[u'rna/data'][:] # <np.array>
        metadata = hf[u'rna/metadata'][:] # <np.array>
        
        p_row = sp_sparse.csr_matrix((data, indices, indptr), shape=(metadata[0],metadata[1]))[random_list,:]

    print(c_row[:20])
    print(p_row)

