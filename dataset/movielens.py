from typing import Union, List, Tuple

import numpy as np
from torch_geometric.data import Data, Dataset
from helper.utils import *


class MovieLensDataset(Dataset):
    def __init__(self, adj_mat, links, labels, h, sample_ratio, max_nodes_per_hop,
                 u_features, v_features, class_values, max_num=None, root='data/movie-lens/ml-latest-small/'):
        super(MovieLensDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(adj_mat)
        self.Acol = SparseColIndexer(adj_mat.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features
        self.class_values = class_values
        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]

    def len(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]
        tmp = subgraph_extraction_labeling(
            (i, j), self.Arow, self.Acol, self.h, self.sample_ratio, self.max_nodes_per_hop,
            self.u_features, self.v_features, self.class_values, g_label
        )
        return construct_pyg_graph(*tmp)