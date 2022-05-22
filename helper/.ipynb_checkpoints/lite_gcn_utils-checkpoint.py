import random
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import softplus
from torch_geometric.utils import structured_negative_sampling


def map_to_index(df):
    old_to_new = {}
    new_to_old = {}
    for new_idx, idx in enumerate(df.index.unique()):
        old_to_new[idx] = new_idx
        new_to_old[new_idx] = idx
    return old_to_new, new_to_old


def get_edges(
    df, edge_col, user_map,
    item_col, item_map, thresh
):
    users = [user_map[idx] for idx in df.index]
    items = [item_map[idx] for idx in df[item_col]]

    ratings = torch.from_numpy(
        df[edge_col].values
    ).view(-1, 1).to(torch.long) >= thresh

    edge_index = [[], []]
    for idx in range(ratings.shape[0]):
        if ratings[idx]:
            edge_index[0].append(users[idx])
            edge_index[1].append(items[idx])

    return torch.tensor(edge_index)


def sample(batch_size, edge_index):
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size,
    )
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices


def bpr_loss(
        user_emb_final, user_emb_0,
        pos_item_emb_final, pos_item_emb_0,
        neg_item_emb_final, neg_item_emb_0,
        lambda_val,
):
    l2_reg = lambda_val * (
        user_emb_0.norm(2).pow(2) +
        pos_item_emb_0.norm(2).pow(2) +
        neg_item_emb_0.norm(2).pow(2)
    )

    pos_scores = torch.mul(user_emb_final, pos_item_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(user_emb_final, neg_item_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)

    loss = -(torch.mean(softplus(pos_scores - neg_scores)) + l2_reg)
    return loss


def get_user_positive_items(edge_index):
    user_pos_items = defaultdict(list)
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        user_pos_items[user].append(item)
    return dict(user_pos_items)


def top_k_metrics(ground_truth, r, k):
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.Tensor(
        [len(ground_truth[i]) for i in range(len(ground_truth))],
    )
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


def get_metrics(model, edge_index, exclude_edge_indices, k):
    user_embedding = model.user_emb.weight
    item_embedding = model.item_emb.weight

    user_embedding = user_embedding / user_embedding.norm(dim=1)[:, None]
    item_embedding = item_embedding / item_embedding.norm(dim=1)[:, None]

    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)
        for user, items in user_pos_items.items():
            rating[[user] * len(items), items] = float('-inf')

    _, top_K_items = torch.topk(rating, k=k)
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users
    ]

    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = top_k_metrics(test_user_pos_items_list, r, k)

    return recall, precision


def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    user_emb_final, user_emb_0, item_emb_final, item_emb_0 = model.forward(
        sparse_edge_index,
    )
    edges = structured_negative_sampling(
        edge_index, contains_neg_self_loops=False,
    )
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    user_emb_final, user_emb_0 = user_emb_final[user_indices], user_emb_0[user_indices]
    pos_item_emb_final, pos_item_emb_0 = item_emb_final[pos_item_indices], item_emb_0[pos_item_indices]
    neg_item_emb_final, neg_item_emb_0 = item_emb_final[neg_item_indices], item_emb_0[neg_item_indices]

    loss = bpr_loss(
        user_emb_final, user_emb_0,
        pos_item_emb_final, pos_item_emb_0,
        neg_item_emb_final, neg_item_emb_0,
        lambda_val,
    ).item()

    recall, precision = get_metrics(
        model, edge_index, exclude_edge_indices, k,
    )

    return loss, recall, precision
