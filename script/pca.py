import math

import numpy as np
import struct
import time
import os
from utils import *
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA

source = '/DATA'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCA projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='gist')
    args = vars(parser.parse_args())
    dataset = args['dataset']

    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{dataset}_base.fvecs')
    query_path = os.path.join(path, f'{dataset}_query.fvecs')
    X = fvecs_read(data_path)
    Q = fvecs_read(query_path)
    N, D = X.shape
    pca = PCA(n_components=D)
    if N < 1000000:
        pca.fit(X)
    else:
        pca.fit(X[:1000000])
    projection_matrix = pca.components_.T
    base = np.dot(X, projection_matrix)
    query = np.dot(Q, projection_matrix)
    mean_ = np.mean(base[:1000000], axis=0)
    var_ = np.var(base[:1000000], axis=0)
    base -= mean_
    query -= mean_
    mean_var = np.vstack((mean_, var_))

    pca_data_path = os.path.join(path, f'{dataset}_proj.fvecs')
    pca_query_path = os.path.join(path, f'{dataset}_query_proj.fvecs')
    matrix_save_path = os.path.join(path, f'{dataset}_pca.fvecs')
    mean_save_path = os.path.join(path, f'{dataset}_mean.fvecs')

    fvecs_write(pca_data_path, base)
    fvecs_write(pca_query_path, query)
    fvecs_write(matrix_save_path, projection_matrix)
    fvecs_write(mean_save_path, mean_var)