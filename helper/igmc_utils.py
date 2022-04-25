import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import scipy.sparse as ssp
from torch_geometric.data import Data, InMemoryDataset


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)


def map_data(data, return_reverse=False):
    uniq = list(set(data))
    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    rev_id_dict = {new: old for new, old in enumerate(sorted(uniq))}
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)

    if return_reverse:
        return data, id_dict, rev_id_dict, n

    return data, id_dict, n


def n_round(arr, n):
    return torch.div(arr, n, rounding_mode='floor') * n


def shuffle_df(df, drop=True):
    rand_idx = np.random.randint(0, df.shape[0], df.shape[0])
    if drop:
        df = df.iloc[rand_idx, :].reset_index(drop=True)
    else:
        df = df.iloc[rand_idx, :]
    return df


def get_nodes(df_ratings, return_reverse=False):
    df_ratings = shuffle_df(df_ratings)
    rated_users = df_ratings.values[:, 0]
    rated_items = df_ratings.values[:, 1]
    ratings = df_ratings.values[:, 2]

    users_tuple = map_data(rated_users, return_reverse)
    items_tuple = map_data(rated_items, return_reverse)

    return users_tuple, items_tuple, ratings


def get_user_features(df_ratings, df_items, genres, genres_map, rated_users_dict, n, sparse=False):
    n_users = len(df_ratings.userId.unique())
    f_len = len(genres_map)
    merged = pd.merge(df_ratings, df_items).drop(labels=['title', 'movieId', 'rating', 'feature'], axis=1)
    grouped = merged.groupby('userId')[genres].mean()
    y = grouped.apply(lambda x: pd.Series((x.nlargest(n))), axis=1).notna().reset_index()

    user_features = np.zeros((n_users, f_len))
    for i in range(y.shape[0]):
        for col in y.columns:
            if y.loc[i, col] is True:
                if col in genres_map:
                    user_features[rated_users_dict[y.loc[i, 'userId']], genres_map[col]] = 1

    if sparse:
        user_features = sp.csr_matrix(user_features)

    return user_features


def get_item_features(df_items, idx_map, sparse=False):
    n_items = df_items.shape[0]
    f_len = len(list(df_items.loc[0, 'feature']))
    item_features = np.zeros((n_items, f_len), dtype=np.float32)

    for movie_id, feature_vec in df_items[['movieId', 'feature']].values.tolist():
        if movie_id in idx_map:
            item_features[idx_map[movie_id], :] = list(feature_vec)

    if sparse:
        item_features = sp.csr_matrix(item_features)

    return item_features


def process_movies(df_movies):
    genres = set()
    lists = df_movies.genres.values
    mp = {}

    def encode(movie_genres):
        res = []
        for idx, genre in enumerate(genres):
            if genre in movie_genres:
                res.append(1)
            else:
                res.append(0)
            mp[genre] = idx
        return res + [''.join([str(i) for i in res])]

    for idx, lis in enumerate(lists):
        for genre in lis.split('|'):
            genres.add(genre)

    genres = sorted(genres)
    df_movies[genres + ['feature']] = df_movies.apply(lambda x: encode(x.genres), 1).values.tolist()

    return df_movies.drop(labels='genres', axis=1), genres, mp


def split(data, rating_dict, ratio=0.8):
    rated_users, rated_items, ratings = data
    n = rated_items.shape[0]
    n_train = int(n * ratio)
    stacked = np.vstack([rated_users, rated_items]).T
    train_pairs_idx = stacked[:n_train]
    test_pairs_idx = stacked[n_train:]

    user_train_idx, item_train_idx = train_pairs_idx.transpose()
    user_test_idx, item_test_idx = test_pairs_idx.transpose()

    labels = np.array([rating_dict[r] for r in ratings], dtype=np.int32)
    train_labels = labels[:n_train]
    test_labels = labels[n_train:]

    return user_train_idx, item_train_idx, user_test_idx, item_test_idx, train_labels, test_labels


def construct_pyg_graph(u, v, r, node_labels, max_node_label, y, node_features):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    edge_type = torch.cat([r, r])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))
    y = torch.FloatTensor([y])
    data = Data(x, edge_index, edge_type=edge_type, y=y)

    if node_features is not None:
        if type(node_features) == list:  # a list of u_feature and v_feature
            u_feature, v_feature = node_features
            data.u_feature = torch.FloatTensor(u_feature).unsqueeze(0)
            data.v_feature = torch.FloatTensor(v_feature).unsqueeze(0)
        else:
            x2 = torch.FloatTensor(node_features)
            data.x = torch.cat([data.x, x2], 1)
    return data


def subgraph_extraction_labeling(ind, Arow, Acol, h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                                 u_features=None, v_features=None, class_values=None,
                                 y=1):
    # extract the h-hop enclosing subgraph around link 'ind'
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = {ind[0]}, {ind[1]}
    u_fringe, v_fringe = {ind[0]}, {ind[1]}
    for dist in range(1, h + 1):
        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio * len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio * len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)
    subgraph = Arow[u_nodes][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0

    # prepare pyg graph constructor input
    u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)
    v += len(u_nodes)
    r = r - 1  # transform r back to rating label
    num_nodes = len(u_nodes) + len(v_nodes)
    node_labels = [x * 2 for x in u_dist] + [x * 2 + 1 for x in v_dist]
    max_node_label = 2 * h + 1
    y = class_values[y]

    # get node features
    if u_features is not None:
        u_features = u_features[u_nodes]
    if v_features is not None:
        v_features = v_features[v_nodes]
    node_features = None

    # if False:
    #     # directly use padded node features
    #     if u_features is not None and v_features is not None:
    #         u_extended = np.concatenate(
    #             [u_features, np.zeros([u_features.shape[0], v_features.shape[1]])], 1
    #         )
    #         v_extended = np.concatenate(
    #             [np.zeros([v_features.shape[0], u_features.shape[1]]), v_features], 1
    #         )
    #         node_features = np.concatenate([u_extended, v_extended], 0)
    # if False:
    #     # use identity features (one-hot encodings of node idxes)
    #     u_ids = one_hot(u_nodes, Arow.shape[0] + Arow.shape[1])
    #     v_ids = one_hot([x + Arow.shape[0] for x in v_nodes], Arow.shape[0] + Arow.shape[1])
    #     node_ids = np.concatenate([u_ids, v_ids], 0)
    #     # node_features = np.concatenate([node_features, node_ids], 1)
    #     node_features = node_ids
    # if True:
    #     # only output node features for the target user and item

    if u_features is not None and v_features is not None:
        node_features = [u_features[0], v_features[0]]

    return u, v, r, node_labels, max_node_label, y, node_features


def neighbors(fringe, A):
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x
